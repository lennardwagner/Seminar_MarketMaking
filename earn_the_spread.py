# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent.agent import BaseAgent
from env.replay import Backtest

import numpy as np
import pandas as pd
import datetime

class Agent(BaseAgent):

    def __init__(self, name, quantity, **kwargs):
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

        # TODO: YOUR IMPLEMENTATION GOES HERE
        self.quantity = quantity

        self.start_time = datetime.time(8, 15)
        self.end_time = datetime.time(8, 25)
        self.calc_time = datetime.time(8, 25, 30)
        self.trading_phase = False
        self.positions_closed = False

        self.total_trades = [0]
        self.buy_trades = []
        self.sell_trades = []

    def on_quote(self, market_id: str, book_state: pd.Series):
        """
        This method is called after a new quote.

        :param market_id:
            str, market identifier
        :param book_state:
            pd.Series, including timestamp, bid/ask price/quantity for 10 levels
        """


        if self.trading_phase:
            limit_buy = book_state["L1-BidPrice"]
            limit_sell = book_state["L1-AskPrice"]
            total_trades = self.market_interface.get_filtered_trades()
            #self.total_trades[0] = len(total_trades)

            if len(total_trades) > self.total_trades[0]:
                # cancel all orders
                for order in self.market_interface.get_filtered_orders(market_id, status="ACTIVE"):
                    print(order.quantity)
                    self.market_interface.cancel_order(order)
                    print("Gecancelt:", order.__str__())
                self.total_trades[0] = len(total_trades)

            if not self.market_interface.get_filtered_orders(market_id, status="ACTIVE"):
                self.market_interface.submit_order(market_id, "buy", self.quantity, limit_buy)
                self.market_interface.submit_order(market_id, "sell", self.quantity, limit_sell)


            '''trades = self.market_interface.get_filtered_trades(market_id, "buy")
            for t in trades:
                print(t.quantity)'''

            # decide is exposure important or not


        '''if not self.market_interface.exposure[market_id] and 
                    not self.market_interface.get_filtered_orders(
                        market_id, status="ACTIVE"):
                self.market_interface.submit_order(market_id, "buy", self.quantity, limit_buy)
                self.market_interface.submit_order(market_id, "sell", self.quantity, limit_sell)'''


    def on_trade(self, market_id: str, trades_state: pd.Series):
        """
        This method is called after a new trade.

        :param market_id:
            str, market identifier
        :param trades_state:
            pd.Series, including timestamp, price, quantity
        """

        # TODO: YOUR IMPLEMENTATION GOES HERE

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

        # TODO: YOUR IMPLEMENTATION GOES HERE
        if timestamp.time() > self.start_time and timestamp.time() < self.end_time:
            self.trading_phase = True
        else:
            self.trading_phase = False


        if timestamp.time() >= self.end_time and not self.positions_closed:

            print("##################################################")
            # close out positions, no exposure over night
            self.positions_closed = True
            for market_id in self.market_interface.market_state_list.keys():
                for order in self.market_interface.get_filtered_orders(market_id, status="ACTIVE"):
                    self.market_interface.cancel_order(order)

                # figure out how many shares the agent is long / short
                shares_bought = 0
                shares_sold = 0

                for trade in self.market_interface.get_filtered_trades(market_id, "buy"):
                    shares_bought += trade.quantity


                for trade in self.market_interface.get_filtered_trades(market_id, "sell"):
                    shares_sold += trade.quantity
                # not sure if sumbit_market_order throws error when quantity is 0
                if shares_bought - shares_sold > 0:
                    self.market_interface.submit_order(market_id, "sell", (shares_bought - shares_sold))
                elif shares_bought - shares_sold < 0:
                    self.market_interface.submit_order(market_id, "buy", abs(shares_bought - shares_sold))
                else:
                    pass

        if timestamp.time() >= self.calc_time and not self.buy_trades:
            for market_id in self.market_interface.market_state_list.keys():
                #print("Aktivierte Aktien sind:", market_id)
                for trade in self.market_interface.get_filtered_trades(market_id, "buy"):
                    self.buy_trades.append(trade)
                for trade in self.market_interface.get_filtered_trades(market_id, "sell"):
                    self.sell_trades.append(trade)




    def summary(self, buy_trades:list, sell_trades:list):

        buy_total_shares = 0
        sell_total_shares = 0
        average_buy = 0

        buy_total_price = 0
        sell_total_price = 0
        average_sell = 0


        for trade in buy_trades:
            buy_total_shares += int(trade.quantity)
            buy_total_price += round(trade.quantity * trade.price, 2)
            average_buy = round(buy_total_price / buy_total_shares, 1)

        for trade in sell_trades:
            sell_total_shares += int(trade.quantity)
            sell_total_price += round(trade.quantity * trade.price, 2)
            average_sell = round(sell_total_price / sell_total_shares, 1)

        return (("buy", buy_total_shares, buy_total_price, average_buy),
                ("sell", sell_total_shares, sell_total_price, average_sell))

if __name__ == "__main__":

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
        quantity=50,
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
    '''backtest.run_episode_generator(identifier_list=identifier_list,
                                   date_start="2021-01-01",
                                   date_end="2021-02-28",
                                   episode_interval=30,
                                   episode_shuffle=True,
                                   episode_buffer=5,
                                   episode_length=30,
                                   num_episodes=10,
                                   )

    # Option 2: run agent against a series of broadcast episodes, that is,
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
    # episode'''
    backtest.run_episode_list(identifier_list=identifier_list,
                              episode_list=[
                                  (
                                  "2021-01-04T08:00:00", "2021-01-04T08:15:00",
                                  "2021-01-04T08:26:00"),

                                  # ...
                              ],
                              )

    '''print(agent.buy_trades)
    for trade in agent.buy_trades:
        print(trade.__str__())'''
    print(agent.summary(agent.buy_trades, agent.sell_trades))


