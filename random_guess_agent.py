# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent.agent import BaseAgent
from env.replay import Backtest

import numpy as np
import pandas as pd
import random
import datetime
random.seed(5)

class Agent(BaseAgent):

    def __init__(self, name, number_of_stocks:int, **kwargs):
        """
        Trading agent implementation.

        The backtest iterates over a set of sources and alerts the trading agent
        whenever a source is updated. These updates are conveyed through method
        calls to, all of which you are expected to implement yourself:

        - on_quote(self, market_id, book_state)
        - on_trade(self, market_id, trade_state)
        - on_time(self, timestamp, timestamp_next)

        In order to interact with a market, the trading agent needs to use the
        market_interface instance available at `self.market_interface` that
        provides the following methods to create and delete orders waiting to
        be executed against the respective market's order book:

        - submit_order(self, market_id, side, quantity, limit=None)
        - cancel_order(self, order)

        Besides, the market_interface implements a set of attributes that may
        be used to monitor trading agent performance:

        - exposure (per market)
        - pnl_realized (per market)
        - pnl_unrealized (per market)
        - exposure_total
        - pnl_realized_total
        - pnl_unrealized_total
        - exposure_left
        - transaction_costs

        The agent may also access attributes of related class instances, using
        the container attributes:

        - order_list -> [<Order>, *]
        - trade_list -> [<Trade>, *]
        - market_state_list -> {<market_id>: <Market>, *}

        For more information, you may list all attributes and methods as well
        as access the docstrings available in the base class using
        `dir(BaseAgent)` and `help(BaseAgent.<method>)`, respectively.

        :param name:
            str, agent name
        """
        super(Agent, self).__init__(name, **kwargs)

        self.number_of_stocks = number_of_stocks

        self.transaction_cost = self.market_interface.transaction_cost_factor
        self.total_transaction_costs = 0

        self.maximum_shares = {}
        self.assigned_guess = {} # fixed through random.seed()

        self.top_level_bid = {}
        self.top_level_ask = {}

        self.total_trades = [0 for i in range(self.number_of_stocks)]

        self.stocks = []
        self.trades = {}

        self.L1_prices = []
        self.minimum_quote_size = {}

        self.time_in_market_so_far = [[] for i in range(self.number_of_stocks)]
        self.current_hour = 8
        self.current_minute = 0
        self.time_checks = [["12:00", False], ["13:00", False],
                            ["14:00", False], ["15:00", False],
                            ["16:00", False]]

        self.last_order_submit = [None for i in range(self.number_of_stocks)]
        self.last_order_executed = [None for i in range(self.number_of_stocks)]

        self.start_time = datetime.time(8, 15)
        self.end_time = datetime.time(16, 15)
        self.calc_time = datetime.time(16, 15, 30)

        self.exposure_limit_total = self.market_interface.exposure_limit
        self.exposure_limit_per = self.exposure_limit_total // self.number_of_stocks

        self.trading_phase = [False for i in range(self.number_of_stocks)]
        self.positions_closed = False

        self.market_id_index = {
            "Adidas": 0,
            "Allianz": 1,
            "BASF": 2,
            "Bayer": 3,
            "BMW": 4,
            "Continental": 5,
            "Covestro": 6,
            "Daimler": 7,
            "DeutscheBank": 8,
            "DeutscheBörse": 9
        }

        self.four_hour_constraint = [False for i in range(self.number_of_stocks)]



    def on_quote(self, market_id: str, book_state: pd.Series):
        """
        This method is called after a new quote.

        :param market_id:
            str, market identifier
        :param book_state:
            pd.Series, including timestamp, bid/ask price/quantity for 10 levels
        """

        if len(self.L1_prices) < self.number_of_stocks:
            temp = [market_id, book_state["L1-BidPrice"]]
            if all(temp[0] not in i for i in self.L1_prices):
                self.L1_prices.append(temp)

        midpoint = (book_state['L1-BidPrice'] + book_state['L1-AskPrice']) / 2
        midpoint = np.round(midpoint, 4)
        self.top_level_bid[market_id] = book_state["L1-BidPrice"]
        self.top_level_ask[market_id] = book_state["L1-AskPrice"]

        if self.maximum_shares:
            if self.maximum_shares[market_id][0] == 0 and \
                    self.maximum_shares[market_id][1] == 0:
                total_shares = self.exposure_limit_per // midpoint
                min_shares = total_shares // 2.5
                max_shares = total_shares - min_shares
                self.maximum_shares[market_id] = [min_shares, max_shares]




        if self.trading_phase[self.market_id_index[market_id]]:
            limit_buy1 = book_state["L1-BidPrice"]
            limit_sell1 = book_state["L1-AskPrice"]
            limit_buy2 = book_state["L2-BidPrice"]
            limit_sell2 = book_state["L2-AskPrice"]

            total_trades = self.market_interface.get_filtered_trades(market_id)

            if abs(self.market_interface.exposure[market_id]) > \
                    self.exposure_limit_per * 0.9:
                self.rebalance(market_id)

            if len(total_trades) > self.total_trades[self.market_id_index[market_id]]:
                # calculate transaction cost of latest trade
                self.total_transaction_costs += round((total_trades[-1].quantity * total_trades[-1].price) * self.transaction_cost, 2)

                # get timestamp
                self.last_order_executed[self.market_id_index[market_id]] = book_state["TIMESTAMP_UTC"]
                time_diff = self.last_order_executed[self.market_id_index[market_id]] - self.last_order_submit[self.market_id_index[market_id]]
                self.time_in_market_so_far[self.market_id_index[market_id]].append(time_diff)

                # cancel all remaining orders
                for order in self.market_interface.get_filtered_orders(market_id, status="ACTIVE"):
                    self.market_interface.cancel_order(order)
                self.total_trades[self.market_id_index[market_id]] = len(total_trades)

            # activates should the agent have traded too little and needs to
            #   meet the 50% rule.
            if self.four_hour_constraint[self.market_id_index[market_id]]:
                if not self.market_interface.get_filtered_orders(market_id, status="ACTIVE"):
                    self.market_interface.submit_order(market_id, "buy",
                                                       self.minimum_quote_size[market_id],
                                                       limit_buy1)
                    self.market_interface.submit_order(market_id, "sell",
                                                       self.minimum_quote_size[market_id],
                                                       limit_sell1)
                    self.last_order_submit[self.market_id_index[market_id]] = \
                        book_state["TIMESTAMP_UTC"]

            # Submit orders
            if not self.market_interface.get_filtered_orders(market_id, status="ACTIVE"):
                if self.assigned_guess[market_id] == "UP":
                    buy_quant = int(self.minimum_quote_size[market_id] * 1.5)
                    sell_quant = self.minimum_quote_size[market_id]
                    self.market_interface.submit_order(market_id, "buy", buy_quant, limit_buy1)
                    self.market_interface.submit_order(market_id, "sell", sell_quant, limit_sell2)
                    self.last_order_submit[self.market_id_index[market_id]] = \
                    book_state["TIMESTAMP_UTC"]
                elif self.assigned_guess[market_id] == "DOWN":
                    sell_quant = int(self.minimum_quote_size[market_id] * 1.5)
                    buy_quant = self.minimum_quote_size[market_id]
                    self.market_interface.submit_order(market_id, "buy", buy_quant, limit_buy2)
                    self.market_interface.submit_order(market_id, "sell", sell_quant, limit_sell1)
                    self.last_order_submit[self.market_id_index[market_id]] = \
                    book_state["TIMESTAMP_UTC"]
                else:
                    print("Error")



    def on_trade(self, market_id: str, trades_state: pd.Series):
        """
        This method is called after a new trade.

        :param market_id:
            str, market identifier
        :param trades_state:
            pd.Series, including timestamp, price, quantity
        """

        pass

    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):
        """
        This method is called with every iteration and provides the timestamps
        for both current and next iteration. The given interval may be used to
        submit orders before a specific point in time.

        :param timestamp:
            pd.Timestamp, timestamp recorded
        :param timestamp_next:
            pd.Timestamp, timestamp recorded in next iteration
        """

        '''if timestamp.time() > self.start_time and timestamp.time() < self.end_time:
            self.trading_phase = True
        else:
            self.trading_phase = False'''

        self.current_hour = timestamp.hour
        self.current_minute = timestamp.minute

        trading_time = timestamp.time() > self.start_time and timestamp.time() < self.end_time
        if trading_time and not all(self.trading_phase):
            self.trading_phase = [True for i in range(self.number_of_stocks)]
        if not trading_time:
            self.trading_phase = [False for i in range(self.number_of_stocks)]


        if not self.minimum_quote_size and len(self.L1_prices) == self.number_of_stocks:
            self.minimum_quote_size = {key:None for key in self.market_interface.market_state_list.keys()}
            temp = list(self.minimum_quote_size)
            #print(temp)
            i = 0
            # necessary for the random order of dictionary
            while not all(self.minimum_quote_size.values()):
                for price in self.L1_prices:
                    if price[0] == temp[i]:
                        self.minimum_quote_size[temp[i]] = np.ceil((10_000) / price[1])
                        i += 1
                        break

        if not self.stocks:
            for market_id in self.market_interface.market_state_list.keys():
                self.stocks.append(market_id)

        # close out build up long / short position
        if timestamp.time() >= self.end_time and not self.positions_closed:
            # close out positions, no exposure over night
            self.positions_closed = True
            for market_id in self.market_interface.market_state_list.keys():
                self.rebalance(market_id)

        # Turn off trading phase if 50% limit has been reached
        for i in range(5):
            if self.current_hour == i + 12 and not self.time_checks[0][1]:
                self.time_checks[0][1] = True
                try:
                    temp = self.time_tracker(self.time_in_market_so_far)
                    for seconds in temp:
                        if seconds >= 15300: # (4 * 60 + 15) * 60
                            self.trading_phase[temp.index(seconds)] = False
                        elif i == 0 and seconds < (3.75 + i) * (15300 / 8): # 8 hours in the market from 8:15 to 16:15
                            self.four_hour_constraint[temp.index(seconds)] = True
                        elif i == 1 and seconds < (3.75 + i) * (15300 / 8):
                            self.four_hour_constraint[temp.index(seconds)] = True
                        elif i == 2 and seconds < (3.75 + i) * (15300 / 8):
                            self.four_hour_constraint[temp.index(seconds)] = True
                        elif i == 3 and seconds < (3.75 + i) * (15300 / 8):
                            self.four_hour_constraint[temp.index(seconds)] = True
                        elif i == 4 and seconds < (3.75 + i) * (15300 / 8):
                            self.four_hour_constraint[temp.index(seconds)] = True

                            # todo potentially rebalance here, after turning
                            #  trading offinstead of after trading time is over
                except AttributeError:
                    pass

        if timestamp.time() >= self.calc_time and not self.trades:
            for market_id in self.market_interface.market_state_list.keys():
                buy_trades = []
                sell_trades = []

                for trade in self.market_interface.get_filtered_trades(market_id, "buy"):
                    buy_trades.append(trade)
                for trade in self.market_interface.get_filtered_trades(market_id, "sell"):
                    sell_trades.append(trade)
                self.trades[market_id] = [buy_trades, sell_trades]

        # Guesses randomly for each stock whether it will go up or down
        if not self.assigned_guess:
            self.assigned_guess = {key: ("UP" if random.randint(1, 2) == 1 else "DOWN") for key in self.market_interface.market_state_list.keys()}
            print(self.assigned_guess)
        # Keep track of the L1 Bid and Ask for each stock
        if not self.top_level_bid:
            self.top_level_bid = {key:None for key in self.market_interface.market_state_list.keys()}
        if not self.top_level_ask:
            self.top_level_ask = {key:None for key in self.market_interface.market_state_list.keys()}

    def summary(self, trades, time_in_market):
        """

        :param trades:
        :return:
        """

        print(self.assigned_guess)

        result = {key: None for key in self.stocks}
        for market_id in result.keys():
            buy_total_shares = 0
            sell_total_shares = 0
            average_buy = 0

            buy_total_price = 0
            sell_total_price = 0
            average_sell = 0

            for trade in trades[market_id][0]:
                buy_total_shares += int(trade.quantity)
                buy_total_price += round(trade.quantity * trade.price, 2)
                average_buy = round(buy_total_price / buy_total_shares, 1)

            for trade in trades[market_id][1]:
                sell_total_shares += int(trade.quantity)
                sell_total_price += round(trade.quantity * trade.price, 2)
                average_sell = round(sell_total_price / sell_total_shares, 1)


            # time in market
            temp = self.time_tracker(self.time_in_market_so_far)

            result[market_id] = ("buy", buy_total_shares, buy_total_price,
                                 average_buy, "sell", sell_total_shares,
                                 sell_total_price, average_sell,
                                 temp[self.market_id_index[market_id]])
        return result

    def rebalance(self, market_id):
        for order in self.market_interface.get_filtered_orders(market_id, status="ACTIVE"):
            self.market_interface.cancel_order(order)
        # figure out how many shares the agent is long / short
        shares_bought = 0
        shares_sold = 0
        for trade in self.market_interface.get_filtered_trades(
                market_id, "buy"):
            shares_bought += trade.quantity
        for trade in self.market_interface.get_filtered_trades(
                market_id, "sell"):
            shares_sold += trade.quantity
        # not sure if sumbit_market_order throws error when quantity is 0
        if shares_bought - shares_sold > 0:
            self.market_interface.submit_order(market_id, "sell", (
                        shares_bought - shares_sold))
        elif shares_bought - shares_sold < 0:
            self.market_interface.submit_order(market_id, "buy",
                                               abs(shares_bought - shares_sold))
        else:
            pass

    def time_tracker(self, timedeltas):
        result = []
        for timedelta in timedeltas:
            seconds = 0
            for i in timedelta:
                seconds += i.components[2] * 60 + i.components[3]
            result.append(seconds)
        return result



if __name__ == "__main__":
    # TODO: SELECT SOURCES. You may delete or comment out the rest.

    identifier_list = [
        # ADIDAS
        "Adidas.BOOK", "Adidas.TRADES",
        # ALLIANZ
        #"Allianz.BOOK", "Allianz.TRADES",
        # BASF
        #"BASF.BOOK", "BASF.TRADES",
        # Bayer
        #"Bayer.BOOK", "Bayer.TRADES",
        # BMW
        #"BMW.BOOK", "BMW.TRADES",
        # Continental
        #"Continental.BOOK", "Continental.TRADES",
        # Covestro
        #"Covestro.BOOK", "Covestro.TRADES",
        # Daimler
        #"Daimler.BOOK", "Daimler.TRADES",
        # Deutsche Bank
        #"DeutscheBank.BOOK", "DeutscheBank.TRADES",
        # DeutscheBörse
        #"DeutscheBörse.BOOK", "DeutscheBörse.TRADES",
    ]

    # TODO: INSTANTIATE AGENT. Please refer to the corresponding file for more
    # information.

    agent = Agent(
        name="test_agent",
        number_of_stocks=(len(identifier_list) // 2),
        # ...
    )

    # TODO: INSTANTIATE BACKTEST. Please refer to the corresponding file for
    # more information.

    backtest = Backtest(
        agent=agent,
    )

    # TODO: RUN BACKTEST. Please refer to the corresponding file for more
    # information.

    # Option 1: run agent against a series of generated episodes, that is,
    # generate episodes with the same episode_buffer and episode_length
    backtest.run_episode_generator(identifier_list=identifier_list,
                                   date_start="2021-02-01",
                                   date_end="2021-02-28",
                                   episode_interval=30,
                                   episode_shuffle=True,
                                   episode_buffer=5,
                                   episode_length=30,
                                   num_episodes=10,
                                   seed=None
                                   )

    """"# Option 2: run agent against a series of broadcast episodes, that is, 
    # broadcast the same timestamps for every date between date_start and 
    # date_end
    backtest.run_episode_broadcast(identifier_list=identifier_list,
                                   date_start="2021-01-01",
                                   date_end="2021-02-28",
                                   time_start_buffer="08:00:00",
                                   time_start="08:30:00",
                                   time_end="16:30:00",
                                   )

    # Option 3: run agent against a series of specified episodes, that is,
    # list a tuple (episode_start_buffer, episode_start, episode_end) for each
    # episode
    backtest.run_episode_list(identifier_list=identifier_list,
                              episode_list=[
                                  (
                                  "2021-02-24T10:00:00", "2021-02-24T10:05:00",
                                  "2021-02-24T10:30:00"),

                              ],
                              )"""
    #print(agent.stocks)
    #print(agent.trades)
    print(agent.summary(agent.trades, agent.time_in_market_so_far))



