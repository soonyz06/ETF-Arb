
class CriticalSolver:
    def __init__(self, traders, lb, ub, hedge_map, fair_values):
        self.traders = traders
        self.lb = lb
        self.ub = ub
        self.hedge_map = hedge_map
        self.fair_values = fair_values

    def _get_critical_values(self, symbol): ##:)
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
        candidates = ask_cum | bid_cum #volume where price changes
        
        lb = self.lb[symbol]
        ub = self.ub[symbol]
        candidates.add(lb)
        candidates.add(ub)
        return [v for v in sorted(candidates) if lb <= v <= ub]

    def _add_hedge_critical(self, assets, all_critical_values):
        hedges = set(self.traders.keys()) - set(assets)
        for hedge in hedges:
            trader = self.traders[hedge]
            hedge_criticals = self._get_critical_values(hedge)
            for i, asset in enumerate(assets):
                conversion = self.hedge_map[asset].get(hedge, 0)
                if conversion == 0:
                    continue
                for candidate in hedge_criticals:
                    q_asset = -candidate / conversion
                    rounded = math.ceil(round(q_asset)) if candidate>0 else math.floor(round(q_asset))
                    if abs(q_asset - rounded) < 1e-8:
                        q_asset = rounded
                    else:
                        continue
                    if self.lb[asset] <= q_asset <= self.ub[asset]:
                        all_critical_values[i].add(q_asset)
        return [sorted(s) for s in all_critical_values]
        
    def _cartesian_product(self, ranges):
        if not ranges:
            yield ()
            return
        for val in ranges[0]:
            for rest in self._cartesian_product(ranges[1:]):
                yield (val, ) + rest

    def _simulate_orders(self, x):
        def _within_pos_limits(x):
            for symbol, qty in x.items():
                trader = self.traders[symbol]
                final_pos = trader.initial_position + qty
                if abs(final_pos) > trader.position_limit:
                    return False
            return True
        
        def _sufficient_liquidity(x):
            for symbol, qty in x.items():
                trader = self.traders[symbol]
                if qty > 0 and qty > sum(trader.ask_orders.values()):
                    return False
                if qty < 0 and abs(qty) > sum(trader.bid_orders.values()):
                    return False
            return True

        if not _within_pos_limits(x) or not _sufficient_liquidity(x): ##interdependence of hedge
            return None, None
        
        orders = []
        total_spread = 0
        for symbol, q in x.items(): 
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
    
    def _calculate_spread(self, x):
        value, orders = self._simulate_orders(x) 
        if value is None:
            return None, None

        total_spread = value
        for symbol, q in x.items():
            fv = self.fair_values.get(symbol, None)
            if fv is None:
                return None, None           
            total_spread += q * fv
        return total_spread, orders

    def parse_composition(self, asset_q, composition):
        part_q = {}
        for symbol, part_map in composition.items():
            for part, qty in part_map.items():
                part_q[part] = part_q.get(part, 0) - qty * asset_q[symbol]
        return part_q
    
    def get_optimal_orders(self, HEDGE=True): #not "optimal" as using asset-only critical values 
        assets = list(self.hedge_map.keys())

        #all_critical_values = [range(self.lb[s], self.ub[s] + 1) for s in assets] #brute force
        all_critical_values = [set(self._get_critical_values(asset)) for asset in assets]
        candidates = set()
        for combo in self._cartesian_product(all_critical_values):
            candidates.add(combo) 

        curr_threshold = 0
        best_orders = []
        for qs in candidates: 
            asset_q = dict(zip(assets, qs))
            hedge_q = self.parse_composition(asset_q, self.hedge_map)
            x = asset_q | hedge_q

            total_spread, orders = self._calculate_spread(x) 
            if total_spread is None:
                continue
            if total_spread > curr_threshold:
                curr_threshold = total_spread
                if not HEDGE:
                    orders = [order for order in orders if order[0] in assets] if orders is not None else None 
                best_orders = orders
        return best_orders

#The only reason u would ever use this is to deal with interdependence, though revised simplex would deal with this better
class MultiTrader:
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict,
                 assets: List[str], hedges: List[str], hedge_map: dict):
        self.assets = assets
        self.hedges = hedges
        self.instruments = assets + hedges
        self.traders = {sym: ProductTrader(sym, state, new_traderData) for sym in self.instruments}

        self.hedge_map = hedge_map
        self.lb, self.ub = self._get_bounds()
    
    def _get_bounds(self): #final position, liquidity and hedge constraints
        lb, ub = {}, {}
        for symbol, trader in self.traders.items():
            lb_pos= -trader.position_limit - trader.initial_position
            ub_pos = trader.position_limit - trader.initial_position
            lb_liq = - sum(trader.bid_orders.values())
            ub_liq = sum(trader.ask_orders.values())
            lb[symbol] = max(lb_pos, lb_liq)
            ub[symbol] = min(ub_pos, ub_liq)
        
        for asset, hedge_map in self.hedge_map.items():
            curr_lb = lb[asset]
            curr_ub = ub[asset]
            for hedge, qty in hedge_map.items():
                if qty == 0:
                    continue
                curr_lb = max(curr_lb, -math.floor(ub[hedge]/qty))
                curr_ub = min(curr_ub, -math.ceil(lb[hedge]/qty))
            lb[asset] = curr_lb
            ub[asset] = curr_ub
        return lb, ub

    def take(self, fair_values, HEDGE): #not global optimal due to interdependence (shared LOB)
        solver = CriticalSolver(self.traders, self.lb, self.ub, self.hedge_map, fair_values)
        orders = solver.get_optimal_orders(HEDGE)
        if not orders:
            return {}
        
        for symbol, price, volume in orders:
            trader = self.traders[symbol]
            if volume > 0:
                trader.buy(price, volume)
            else:
                trader.sell(price, -volume)
        
        return   

    def get_orders(self):
        return {}


class BasketTrader2(MultiTrader):
    def __init__(self, symbol: str, state: TradingState, new_traderData: dict):
        super().__init__(symbol, state, new_traderData, 
                         assets=ETF_ASSETS, hedges=ETF_HEDGES, hedge_map=ETF_COMPOSITION)
        self.composition = ETF_COMPOSITION
    
    def calculate_fair_values(self): 
        fair_values = {}
        for basket in self.assets:
            composition = self.composition.get(basket, {}) #:)
            nav = sum(self.traders[part].mid_price * qty for part, qty in composition.items())
            fair_values[basket]= nav
        for hedge in self.hedges:
            fair_values[hedge] = self.traders[hedge].mid_price

        self.traders[hedge].new_traderData["fv"] = fair_values
        return fair_values

    def get_orders(self):
        fair_values = self.calculate_fair_values()
        if fair_values is None:
            return {}
        #fair_values = {symbol: 0 for symbol in self.instruments}
        self.take(fair_values, HEDGE=False)
        return {trader.symbol: trader.orders for trader in self.traders.values()}  
