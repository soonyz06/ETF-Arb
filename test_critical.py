
class CandidateConstructor:
    def __init__(self, traders, limits, max_buy, max_sell, hedge_map):
        self.traders = traders
        self.limits = limits
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.hedge_map = hedge_map
        
    def _get_critical_values(self, symbol): 
        def _get_cumulative_volume(orders: dict, k: int = 1):
            cum = {0}
            counter = 0 
            for volume in orders.values():
                counter += volume
                cum.add(counter*k)
            return cum
          
        trader = self.traders[symbol]
    
        ask_cum = _get_cumulative_volume(trader.ask_orders)
        bid_cum = _get_cumulative_volume(trader.bid_orders, k=-1)
        candidates = ask_cum | bid_cum #volumes where prices change (breakpoints)

        low_pos = -self.limits[symbol] - trader.initial_position
        high_pos = self.limits[symbol] - trader.initial_position
        low_liq = -self.max_sell.get(symbol, 0)
        high_liq = self.max_buy.get(symbol, 0)
        low = max(low_pos, low_liq)
        high = min(high_pos, high_liq) #position limit and liquidity constraints

        candidates.add(low) #
        candidates.add(high)
        return [v for v in sorted(candidates) if low <= v <= high]
    
    def _solve_linear_equations(self, equations): #Ax=B 
        #solve to find q that satisfies all equations (critical values)
        A = np.array([eq[0] for eq in equations], dtype=float)
        b = np.array([eq[1] for eq in equations], dtype=float)
        try:
            x = np.linalg.solve(A, b)
            return x.tolist()
        except np.linalg.LinAlgError:
            return None
        
    def _add_neighbours(self, point): #3^n Ints
        n = len(point)
        offsets = [-1, 0, 1]
        result = []
        def recurse(i, current):
            if i == n:
                result.append(tuple(current))
                return
            for d in offsets:
                recurse(i + 1, current + [point[i] + d])
        recurse(0, [])
        return result

    def _cartesian_product(self, ranges):
        if not ranges:
            yield ()
            return 
        for val in ranges[0]:
            for rest in self._cartesian_product(ranges[1:]):
                yield (val, ) + rest   

    def get_candidate_points_fast(self, assets, candidates=None):
        if candidates is None: candidates = set()
        per_asset_critical_values = [self._get_critical_values(asset) for asset in assets]
        for combo in self._cartesian_product(per_asset_critical_values):
            candidates.add(combo)
        return candidates
    
    def get_candidate_points(self, assets):
        candidates = set()
        """
        n = len(assets)
        idx = {symbol: i for i, symbol in enumerate(assets)}

        hedge_coeff = {} 
        for symbol, hedge_map in self.hedge_map.items(): 
            i = idx[symbol]
            for hedge, qty in hedge_map.items():
                if hedge not in hedge_coeff:
                    hedge_coeff[hedge] = [0.0]*n
                hedge_coeff[hedge][i] -= qty #expresse hedge equations in terms of assets (list[asset_q])

        #Critical Values
        equations = []
        for symbol, i in idx.items():
            for C in self._get_critical_values(symbol):
                coeff = [0.0]*n
                coeff[i] = 1.0
                equations.append((coeff, float(C))) #as optimal spread has to occur at a critical value (C)
        
        for hedge, coeff in hedge_coeff.items():
            if all(abs(c) < 1e-9 for c in coeff):
                continue
            for C in self._get_critical_values(hedge):
                equations.append((coeff[:], float(C)))
        
        def combine(start, current): #vertex enumeration
            if len(current) == n:
                qs = self._solve_linear_equations(current)
                if qs is None:
                    return
                point = tuple(int(round(q)) for q in qs)
                for nb in self._add_neighbours(point):
                    candidates.add(nb)
                return
            for i in range(start, len(equations)):
                combine(i+1, current+[equations[i]])
        combine(0, []) 
        """
        candidates = self.get_candidate_points_fast(assets, candidates)
        return candidates


class ComplexTrader:
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict, all_symbols: list[str], hedges: dict):
        self.traders = {s: ProductTrader(s, state, new_traderData) for s in all_symbols}
        self.hedges = hedges
        self.limits, self.max_buy, self.max_sell = self._compute_limits()
        self.constructor = CandidateConstructor(self.traders, self.limits, self.max_buy, self.max_sell, self.hedges)
        self.settings = {}

    def _compute_limits(self):
        pos_limits, max_buy, max_sell = {}, {}, {}
        for symbol, trader in self.traders.items():
            pos_limits[symbol] = trader.position_limit
            max_buy[symbol] = sum(trader.ask_orders.values())
            max_sell[symbol] = sum(trader.bid_orders.values())
        return pos_limits, max_buy, max_sell
    
    def _calculate_hedges(self, asset_q):
        hedge_q = self.parse_composition(asset_q, self.hedges)
        return asset_q | hedge_q

    def _within_pos_limits(self, net_q):
        for symbol, qty in net_q.items():
            final_pos = self.traders[symbol].initial_position + qty
            if abs(final_pos)>self.limits[symbol]:
                return False
        return True
    
    def _sufficient_liquidity(self, net_q):
        for symbol, qty in net_q.items():
            if qty>0 and qty>self.max_buy[symbol]:
                return False
            elif qty<0 and abs(qty)>self.max_sell[symbol]:
                return False
        return True


    def parse_composition(self, asset_q, composition):
        part_q = {}
        for symbol, part_map in composition.items():
            for part, qty in part_map.items():
                part_q[part] = part_q.get(part, 0) - qty * asset_q[symbol]
        return part_q
    
    def simulate_orders(self, net_q):
        orders = []
        total_spread = 0
        for symbol, q in net_q.items(): 
            if q==0:
                continue

            trader = self.traders[symbol]
            remaining = abs(q)

            if q > 0:
                for ask, volume in trader.ask_orders.items():
                    size =  min(remaining, volume)
                    if size > 0:
                        orders.append((symbol, ask, size))
                        total_spread -= ask * size
                        remaining -= size
                    if remaining == 0:
                        break
                if remaining > 0:
                    return None, None
                    
            elif q < 0:
                for bid, volume in trader.bid_orders.items():
                    size = min(remaining, volume)
                    if size > 0:
                        orders.append((symbol, bid, -size))
                        total_spread += bid * size
                        remaining -= size
                    if remaining == 0:
                        break
                if remaining > 0:
                    return None, None
        return total_spread, orders
    
    def calculate_spread(self, asset_q, fair_values):
        value, orders = self.simulate_orders(asset_q)
        if value is None:
            return None, None
        
        total_spread = value
        for symbol, q in asset_q.items():
            total_spread += q * fair_values.get(symbol, 0)
        return total_spread, orders

    def get_orders(self):
        assets = self.hedges.keys()
        candidate_points = self.constructor.get_candidate_points(assets)

        curr_threshold = 0
        best_orders = []
        best_asset_q = None
        best_net_q = None
        for qs in candidate_points: #Hedging constraint, Position limit, Liquidity and Spread constraints
            asset_q = dict(zip(assets, qs))
            net_q = self._calculate_hedges(asset_q) 
            if not self._within_pos_limits(net_q) or not self._sufficient_liquidity(net_q): 
                continue

            total_spread, fills = self.calculate_spread(asset_q, **self.settings) ##fix all naming, slippage, combine with simple
            if total_spread is None:
                continue

            if total_spread > curr_threshold:
                curr_threshold = total_spread
                best_orders = fills
                best_asset_q = asset_q
                best_net_q = net_q
        if not best_orders:
            return {}
        
        hedge_q = {k: v for k, v in best_net_q.items() if k not in best_asset_q}
        _, hedge_orders = self.simulate_orders(hedge_q)
        if hedge_orders is None:
            return {}
        
        all_orders = best_orders + hedge_orders
        for symbol, price, volume in all_orders:
            trader = self.traders[symbol]
            if volume > 0:
                trader.buy(price, volume)
            else:
                trader.sell(price, -volume)
        return {trader.symbol: trader.orders for trader in self.traders.values()}    

class BasketTrader(ComplexTrader):
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict):
        super().__init__(symbol, state, new_traderData, all_symbols=ETF_SYMBOLS, hedges=ETF_HEDGE_MAP)
        self.composition = ETF_COMPOSITION
    
    #@override
    def calculate_spread(self, asset_q): 
        assets = asset_q.keys()
        total_spread, orders = self.simulate_orders(net_q = (asset_q | self.parse_composition(asset_q, self.composition)))
        asset_orders = [order for order in orders if order[0] in assets] if orders is not None else None
        return total_spread, asset_orders


    
