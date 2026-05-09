from typing import List, Dict, Tuple, Any
import numpy as np


class RevisedSimplexSolver:
    """
    Goal: Minimise c'x, subject to constraint Ax=b 

    1)Bx=b to get a vertex of feasible region (as B is a basis)
    2)Move to the neighbouring vertex (swap entering (rc) and leaving (d))
    3)Repeat until no improvement

    obj = c @ x 
    x = self._find_vertex(B, self.b)
    rc = c - self._find_vertex(B.T, c[basis]) @ A
    d = self._find_vertex(B, A[:, entering])   
    """
    def __init__(self, c, ub_liq, A_pos=None, b_pos=None, A_hedge=None, b_hedge=None, minimise=True, tol=1e-6, max_iter=2000):
        """
        x: List[volume traded per order-book level]
        c: List[-profit per volume]
        """
        self.minimise = minimise
        self.tol = tol
        self.max_iter = max_iter
        self.n = len(c)
        self.n_slack_pos = 0
        self.n_slack_liq = 0
    
        A_rows, b_rows = self._initialise_constraints(ub_liq, A_pos, b_pos, A_hedge, b_hedge)
        self.A_full = np.vstack(A_rows).astype(float) 
        self.b_full = np.concatenate(b_rows).astype(float)
        self.m = len(self.b_full) 
        self.c_full = np.concatenate([c, np.zeros(self.n_slack_pos + self.n_slack_liq)]) #convert inequalities to equalities using slack variables 

    def _initialise_constraints(self, ub_liq, A_pos, b_pos, A_hedge, b_hedge): #Ax=b
        """
        A_hedge: Row: each hedge instruments expressed in terms of each instrument 
        b_hedge: Ax=B=0, delta neutral with respect to all hedges
        A_pos: [[+_get_direction], [-_get_direction], ...]
        b_pos: [max_buy, max_sell, ...]
        ub_liq: [max_volume]
        """

        n_vars = self.n
        n_slack_pos = len(b_pos) if A_pos is not None and b_pos is not None else 0
        n_slack_liq = n_vars if ub_liq is not None else 0
        self.n_slack_pos = n_slack_pos
        self.n_slack_liq = n_slack_liq
        
        A_rows = []
        b_rows = []

        #Liquidity (0 ≤ x ≤ ub_liq)
        if ub_liq is not None: 
            I_x = np.eye(n_vars)                                 
            zero_pos = np.zeros((n_vars, n_slack_pos))           
            I_liq = np.eye(n_vars)                              
            full_A = np.hstack([I_x, zero_pos, I_liq])
            A_rows.append(full_A)
            b_rows.append(np.array(ub_liq))

        #Final Position Limits (A_pos @ x ≤ b_pos)
        if A_pos is not None and b_pos is not None:
            I_pos = np.eye(n_slack_pos)                          
            zero_liq = np.zeros((n_slack_pos, n_slack_liq))      
            full_A = np.hstack([A_pos, I_pos, zero_liq])
            A_rows.append(full_A)
            b_rows.append(np.array(b_pos))

        #Hedging Constraint (A_hedge @ x = b_hedge)
        if A_hedge is not None and b_hedge is not None:
            zero_pos = np.zeros((A_hedge.shape[0], n_slack_pos))
            zero_liq = np.zeros((A_hedge.shape[0], n_slack_liq))
            full_A = np.hstack([A_hedge, zero_pos, zero_liq])
            A_rows.append(full_A)
            b_rows.append(np.array(b_hedge))
        
        #total rows (number of constraints): (number of order‑book levels) + 2*len(instruments) + len(hedges)  
        return A_rows, b_rows

    def _find_vertex(self, B, b):
        try:
            return np.linalg.solve(B, b) #faster
        except np.linalg.LinAlgError:
            x, residuals, rank, s = np.linalg.lstsq(B, b, rcond=self.tol) #works for singular matrices
            if rank < B.shape[0]:
                raise ValueError("Basis matrix singular")
            return x

    def _standard_rule(self, rc, basis): #cycling risk in degenerate problem (multiple BFS represent same vertex)
        total_vars = len(rc)
        entering = None
        if self.minimise:
            best_val = np.inf           
            for j in range(total_vars):
                if j in basis:
                    continue
                if rc[j] < best_val - self.tol:
                    best_val = rc[j]
                    entering = j
            if best_val >= -self.tol:   
                return None
        else:
            best_val = -np.inf          
            for j in range(total_vars):
                if j in basis:
                    continue
                if rc[j] > best_val + self.tol:
                    best_val = rc[j]
                    entering = j
            if best_val <= self.tol:    
                return None
        return entering

    def _blands_rule(self, rc, basis): #gurantees termination, but slower
        total_vars = len(rc)
        for j in range(total_vars): #Bland's rule to prevent cycling
            if j in basis:
                continue
            if self.minimise:
                if rc[j] < -self.tol:
                    return j
            else:
                if rc[j] > self.tol:
                    return j
        return None
                    
    def _ratio_test(self, d, basis, x_B): #find basic variable that reaches 0 first
        leaving = None
        min_ratio = np.inf
        for idx, bvar in enumerate(basis): #Ratio test
            if d[idx] > self.tol:
                ratio = x_B[idx] / d[idx]
                if ratio < min_ratio - self.tol:
                    min_ratio = ratio
                    leaving = idx
                elif abs(ratio - min_ratio) <= self.tol:
                    if bvar < basis[leaving]:
                        leaving = idx
        return leaving

    def _simplex(self, c, A, b, initial_basis): #Finds optimal basis and BFS for given inputs
        m  = len(b)
        n = len(c)

        basis = initial_basis.copy()
        B = A[:, basis]
        x_B = self._find_vertex(B, b)

        for it in range(self.max_iter): #Find next neighbouring vertex
            rc = c - self._find_vertex(B.T, c[basis]) @ A
            entering = self._standard_rule(rc, basis)
            if entering is None: #Optimal BFS (No improvement)
                break

            d = self._find_vertex(B, A[:, entering])
            leaving = self._ratio_test(d, basis, x_B)
            if leaving is None:
                raise ValueError("Unbounded?") 
            
            basis[leaving] = entering #Update basis and find new BFS
            B = A[:, basis]
            x_B = self._find_vertex(B, b)
        return basis, x_B

    def _initial_basis(self):
        n_full = len(self.c_full) 
        m = self.m 

        #minimise Σ(artificial variable) subject to Ax + Ia = b constraint
        A_phase = np.hstack([self.A_full, np.eye(m)]) 
        c_phase = np.zeros(n_full + m)
        c_phase[n_full:] = 1.0 
        initial_basis = list(range(n_full, n_full + m))
    
        basis, x_B = self._simplex(c_phase, A_phase, self.b_full, initial_basis)
        for i, idx in enumerate(basis):
            if idx >= n_full and x_B[i] > self.tol:
                raise ValueError(f"Infeasible problem (artificial {idx} = {x_B[i]})")
        
        basis = [b for b in basis if b < n_full]
        if len(basis) < m:
            for j in range(n_full):
                if j not in basis:
                    trial = basis + [j]
                    B_trial = self.A_full[:, trial]
                    if np.linalg.matrix_rank(B_trial) > len(basis):
                        basis = trial
                        if len(basis) == m:
                            break
        if len(basis) != m: 
            raise ValueError("Cannot form full basis")
        return basis
    
    def solve(self):
        #Minimise c'x subject to Ax=b constraint
        basis = self._initial_basis()
        basis, x_B = self._simplex(self.c_full, self.A_full, self.b_full, basis)
        
        n_full = len(self.c_full) 
        x = np.zeros(n_full)
        x[basis] = x_B
        obj = self.c_full @ x
        orig_len = len(self.c_full) - self.n_slack_pos - self.n_slack_liq 
        return x[:orig_len], obj

class SimplexTrader:
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict, 
                 assets: List[str], hedges: List[str], hedge_map: dict, position_limits: dict):
        self.assets = assets
        self.hedges = hedges
        self.hedge_map = hedge_map
        self.instruments = self.assets + self.hedges
        self.traders = {s: ProductTrader(s, state, new_traderData) for s in self.instruments}
        self.position_limits = position_limits
        
        self.fair_values = {}
        self.x_info = []
    
    def _get_direction(self, symbol): 
        row = np.zeros(len(self.x_info))
        for j, (v_sym, direction, _, _) in enumerate(self.x_info):
            if v_sym == symbol:
                row[j] = direction
        return row
    
    def _parse_lobs(self):
        x_info = []
        c = []
        ub_liq = []

        for symbol in self.instruments:
            trader = self.traders[symbol]
            fair_value = self.fair_values.get(symbol, None) if symbol in self.assets else trader.mid_price
            if fair_value is None:
                raise ValueError(f"Missing fair value for asset {symbol}")
            
            for bid, volume in trader.bid_orders.items(): 
                x_info.append((symbol, -1, bid, volume))
                profit = bid - fair_value
                c.append(-profit)
                ub_liq.append(volume)
            for ask, volume in trader.ask_orders.items():
                x_info.append((symbol, 1, ask, volume))
                profit = fair_value - ask
                c.append(-profit)
                ub_liq.append(volume)

        self.x_info = x_info
        c = np.array(c)
        ub_liq = np.array(ub_liq) if ub_liq else None
        return c, ub_liq

    def _parse_final_pos(self):
        max_buy = {}
        max_sell = {}
        for symbol in self.instruments:
            net_position = self.traders[symbol].initial_position
            L = self.position_limits.get(symbol, 0)
            max_buy[symbol] = L - net_position
            max_sell[symbol] = L + net_position

        A_pos = []
        b_pos = []
        for symbol in self.instruments:
            row = self._get_direction(symbol)
            A_pos.append(row)
            b_pos.append(max_buy[symbol])   # q <= max_buy
            A_pos.append(-row)
            b_pos.append(max_sell[symbol])  # -q <= max_sell
        
        A_pos = np.array(A_pos) if A_pos else None
        b_pos = np.array(b_pos) if b_pos else None
        return A_pos, b_pos

    def _parse_hedges(self):
        A_hedge = []
        b_hedge = []
        for hedge in self.hedges:
            row = 1 * self._get_direction(hedge) #add own q
            for asset in self.assets:
                qty = self.hedge_map.get(asset, {}).get(hedge, 0)
                row += qty * self._get_direction(asset)
            A_hedge.append(row)
            b_hedge.append(0.0)
        
        A_hedge = np.array(A_hedge) if A_hedge else None
        b_hedge = np.array(b_hedge) if b_hedge else None
        return A_hedge, b_hedge

    def get_orders(self):
        c, ub_liq = self._parse_lobs()
        A_pos, b_pos = self._parse_final_pos()
        A_hedge, b_hedge = self._parse_hedges()

        solver = RevisedSimplexSolver(
            c=c, 
            ub_liq=ub_liq, 
            A_pos=A_pos, 
            b_pos=b_pos,
            A_hedge=A_hedge, 
            b_hedge=b_hedge, 
            minimise=True,
            tol=1e-8, 
            max_iter=2000
        )
        x, obj = solver.solve()
        
        if obj >= 0:
            return {}

        orders = {symbol: [] for symbol in self.instruments}
        for j, (symbol, direction, price, _) in enumerate(self.x_info):
            size = x[j]
            if size <= 1e-8:
                continue
            orders[symbol].append((price, size, direction))
        
        for symbol, order_list in orders.items():
            trader = self.traders[symbol]
            for price, size, direction in order_list:
                if direction > 0:
                    trader.buy(price, size)
                elif direction < 0:
                    trader.sell(price, -size)
        return {trader.symbol: trader.orders for trader in self.traders.values()}

class BasketTrader(SimplexTrader):
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict):
        super().__init__(symbol, state, new_traderData, 
                         assets=ETF_ASSETS, hedges=ETF_HEDGES, 
                         hedge_map=ETF_HEDGE_MAP, position_limits=POS_LIMITS)
        self.composition = ETF_COMPOSITION
        self.fair_values = {
            asset: sum(trader.mid_price * self.composition[asset].get(part, 0) for part, trader in self.traders.items())
            for asset in self.assets
        }
        self.traders[symbol].new_traderData["fv"] = self.fair_values

        
