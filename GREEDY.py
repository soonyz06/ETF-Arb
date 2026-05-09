class BasketTrader: #add B1 = B2 (inter)
    def __init__(self, symbol, state, new_traderData):
        self.basket_traders = {b: ProductTrader(b, state, new_traderData) for b in ETF_BASKETS}
        self.part_traders = {p: ProductTrader(p, state, new_traderData) for p in ETF_PARTS}

    def _update_premium(self, trader: ProductTrader, new_obs, alpha=0.1, key="premium"):
        prior = trader.last_traderData.get(key, new_obs)
        posterior = alpha*new_obs + (1-alpha)*prior
        trader.new_traderData[key] = posterior
        return posterior

    def _check_order_within_pos_limit(self, order: dict, potential_positions: dict): #{symbol: {p: q}}, {symbol: initial_position}
        for symbol, p_map in order.items():
            net_position = potential_positions.get(symbol, 0) + sum(p_map.values())
            if abs(net_position) > POS_LIMITS[symbol]:
                return False
        return True
    
    def _yield_top_unit(self, book: dict, composition: dict): #generate top unit subject to liquidity and pos limit constraints
        temp_book = {s: {p: v for p, v in o.items()} for s, o in book.items()}

        while True:
            for symbol in temp_book.keys(): 
                vol_per_unit = composition.get(symbol, 1)  #liquidity constraint (not enough for 1 unit), pos constraint done outside (break)
                if sum(temp_book[symbol].values()) < vol_per_unit:
                    return 
            
            top_layer = {}
            for symbol in temp_book.keys():
                needed_vol = composition.get(symbol, 1)  
                prices_taken = {}
                while needed_vol>0: 
                    price = next(iter(temp_book[symbol]))
                    available_vol = temp_book[symbol][price]
                    take = min(needed_vol, available_vol) 
                    prices_taken[price]=take 

                    temp_book[symbol][price] -= take 
                    needed_vol -= take
                    if temp_book[symbol][price] <=0: temp_book[symbol].pop(price)
                top_layer[symbol] = prices_taken
            yield top_layer #{symbol: {p: v}}    
        
    def _get_possible_units(self, basket_symbol: str):  
        basket_trader = self.basket_traders[basket_symbol]
        potential_positions = self.current_positions.copy()
        composition = ETF_COMPOSITION[basket_symbol]
        
        #Get Premium and Thresholds
        nav = sum(self.part_traders[symbol].mid_price * qty for symbol, qty in composition.items())
        basket_trader.new_traderData["nav"] = int(nav)
        raw_spread = basket_trader.mid_price - nav
        premium = self._update_premium(basket_trader, raw_spread)

        best_spread = (basket_trader.best_ask-basket_trader.best_bid)
        threshold = ETF_THRESHOLDS.get(basket_symbol, best_spread)

        #Generate custom book (reversed cross-directional)
        basket_book_base = {basket_symbol: basket_trader.buy_orders} #short bid
        parts_book_base = {basket_symbol: basket_trader.sell_orders} #long ask
        for part_symbol in composition.keys():
            part_trader = self.part_traders[part_symbol]
            basket_book_base[part_symbol] = part_trader.sell_orders 
            parts_book_base[part_symbol] = part_trader.buy_orders 

        #Calculate spreads at each top_layer :{symbol: price}
        orders = {}
        for top_layer in self._yield_top_unit(basket_book_base, composition): 
            basket_info = top_layer.pop(basket_symbol)
            basket_fair_value = sum(p*q for p, q in basket_info.items()) + premium
            parts_fair_value = 0
            for p_map in top_layer.values(): parts_fair_value += sum(p*q for p, q in p_map.items())

            spread = basket_fair_value - parts_fair_value
            if spread > threshold: #Short basket, Long parts
                order = {basket_symbol: {p: -q for p, q in basket_info.items()}}
                for p_sym, p_map in top_layer.items(): order[p_sym] = p_map 

                if not self._check_order_within_pos_limit(order, potential_positions):
                    break
                if spread not in orders: orders[spread] = []
                orders[spread].append(order)
                for symbol, p_map in order.items():
                    potential_positions[symbol] = potential_positions.get(symbol, 0) + sum(p_map.values())
            else:
                break
        
        for top_layer in self._yield_top_unit(parts_book_base, composition): 
            basket_info = top_layer.pop(basket_symbol)
            basket_fair_value = sum(p*q for p, q in basket_info.items()) + premium
            parts_fair_value = 0
            for p_map in top_layer.values():
                parts_fair_value += sum(p*q for p, q in p_map.items())
            
            spread = parts_fair_value - basket_fair_value
            if spread > threshold: #Short parts, Long basket
                order = {basket_symbol: basket_info} 
                for p_sym, p_map in top_layer.items(): order[p_sym] = {p: -q for p, q in p_map.items()} 

                if not self._check_order_within_pos_limit(order, potential_positions):
                    break
                if spread not in orders: orders[spread] = []
                orders[spread].append(order)
                for symbol, p_map in order.items():
                    potential_positions[symbol] = potential_positions.get(symbol, 0) + sum(p_map.values())
            else:
                break

        basket_trader.new_traderData[basket_symbol] = potential_positions #Debug
        return orders        

    def _flatten_units(self, final_trades: list): #[{symbol: {p: q}}] to {symbol: {p: q}}        
            flattened_trades = {} 
            for trade in final_trades:
                for symbol, p_map in trade.items():
                    if symbol not in flattened_trades:
                        flattened_trades[symbol] = {}
                    for price, volume in p_map.items():
                        flattened_trades[symbol][price] = flattened_trades[symbol].get(price, 0) + volume
            return flattened_trades 

    def get_orders(self):  
        all_traders = self.basket_traders | self.part_traders
        self.current_positions = {symbol: trader.initial_position for symbol, trader in all_traders.items()}

        #Get Units (across all baskets) 
        all_orders = {}
        for basket_name in ETF_COMPOSITION.keys():
            basket_orders = self._get_possible_units(basket_name) #liquidity constraint
            for spread, order in basket_orders.items():
                all_orders[spread] = all_orders.get(spread, []) + order #{spread: list[Unit]}
        
        #Greedy Selection
        selected_units = [] 
        potential_positions = self.current_positions.copy()
        for spread in sorted(all_orders.keys(), reverse=True): #greedy 
            for unit in all_orders[spread]:
                if not self._check_order_within_pos_limit(unit, potential_positions): #pos limit constraint
                    continue
                selected_units.append(unit)
                for symbol, p_map in unit.items():
                    potential_positions[symbol] = potential_positions.get(symbol, 0) + sum(p_map.values()) #List[Unit]
        selected_units = self._flatten_units(selected_units) 

        #Execute Units
        for symbol, p_map in selected_units.items(): 
            current_trader = all_traders[symbol]
            #if symbol not in ETF_BASKETS: continue 
            for price, volume in p_map.items():
                if volume > 0:
                    current_trader.buy(price, volume)
                elif volume < 0:
                    current_trader.sell(price, volume)
        return {symbol: trader.orders for symbol, trader in all_traders.items()} 
                
