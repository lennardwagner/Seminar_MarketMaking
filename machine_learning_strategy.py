# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from agent.agent import BaseAgent
from env.replay import Backtest
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import sys
#pip install openpyxl


##global result storage:
result_matrix = pd.DataFrame()
result_per_Share = pd.DataFrame()
result_exposure = pd.DataFrame()

model_Adidas = tf.keras.models.load_model("./ML_models/Adidas_60tik")
model_Allianz = tf.keras.models.load_model("./ML_models/Allianz_60tik")
model_BASF = tf.keras.models.load_model("./ML_models/BASF_60tik")
model_Bayer = tf.keras.models.load_model("./ML_models/Bayer_60tik")
model_BMW = tf.keras.models.load_model("./ML_models/BMW_60tik")
model_Continental = tf.keras.models.load_model("./ML_models/Continental_60tik")
model_Covestro = tf.keras.models.load_model("./ML_models/Covestro_60tik")
model_Daimler = tf.keras.models.load_model("./ML_models/Daimler_60tik")
model_DeutscheBank = tf.keras.models.load_model("./ML_models/DeutscheBank_60tik")
model_DeutscheBoerse = tf.keras.models.load_model("./ML_models/DeutscheBörse_60tik")

class SimpleAgent(BaseAgent):
    def __init__(self, name: str, number_of_stocks: int):

        super(SimpleAgent, self).__init__(name)
        self.number_of_stocks = number_of_stocks
        # static attributes from arguments

        self.max_spread_quoted = 0.002
        self.min_vol_quote = 10000

        # dynamic attributes
        self.spread_market = {} #dict that captures the current spread traded
        self.max_quant_ask = {} #dict, that captures the maximal tradeable quantity of the ask side
        self.max_quant_bid = {} #dict, that captures the maximal tradeable quantity of the bid side
        self.min_quant_ask = {} #dict, that captures the minimal tradeable quantity of the ask side
        self.min_quant_bid = {} #dict, that captures the minimal tradeable quantity of the bid side


        ##### Evaluation, dynamic Variables###
        self.titm = {}  # Time in the market dictionary
        self.sess_length = datetime.timedelta()  # Time the whole Backtestingsession took
        self.VWAP = {}  # stores the VWAP realized of the agent per share
        self.VWAP_Market = {}  # stores the Market VWAP realized
        self.D_V_M = {}  #accumalates dollar volume traded per share in the back testing session
        self.T_V_M = {} #accumalates volume traded per share in the back testing session
        self.VWAP_buy = {}
        self.VWAP_sell = {}
        self.VWAP_Score = {} #stores the VWAP Score of market VWAP and share VWAP
        self.trigger_storage_stop_loss = {}
        self.trigger_storage_take_prof = {}
        self.exposure_check = 0
        self.exposure_stor = [0]
        self.iterer = datetime.timedelta(seconds=30)
        self.milliseconds = datetime.timedelta(milliseconds=100)


        self.last_60_ticks = [[] for i in range(self.number_of_stocks)]
        self.total_trades = [0 for i in range(self.number_of_stocks)]
        self.cancel_all = False
        self.trading_phase = False
        self.rebalance_orders_submitted = False
        self.assigned_guess = {}
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
        
        

    def on_quote(self, market_id: str, book_state: pd.Series):
        """# cancel remaining orders
        #if market_id == "BASF":
            #print(self.total_trades[self.market_id_index[market_id]])
            #print("len total_trades:", len(total_trades))"""
        
        midpoint = (book_state['L1-BidPrice'] + book_state['L1-AskPrice']) / 2
        self.spread_market[market_id] = (book_state['L1-AskPrice'] - book_state['L1-BidPrice']) / midpoint
        self.min_quant_ask[market_id] = math.ceil(self.min_vol_quote / book_state['L1-BidPrice'])
        self.min_quant_bid[market_id] = math.ceil(self.min_vol_quote / book_state['L1-AskPrice'])
        #self.max_quant_ask[market_id] = max(math.floor(0.95 * self.market_interface.exposure_left / book_state['L1-BidPrice']),0)
        #self.max_quant_bid[market_id] = max(math.floor(0.95 * self.market_interface.exposure_left / book_state['L1-AskPrice']),0)


        """trades_buy = self.market_interface.get_filtered_trades(market_id, side="buy")
        trades_sell = self.market_interface.get_filtered_trades(market_id, side="sell")
        # quantity per market
        quantity_buy = sum(t.quantity for t in trades_buy)
        quantity_sell = sum(t.quantity for t in trades_sell)
        quantity_unreal = quantity_buy - quantity_sell #>0 long, <0, short"""
        
        assigned_guess = {}
        assigned_guess[market_id] = ""
        
        #get last 60 ticks per stock
        if len(self.last_60_ticks[self.market_id_index[market_id]]) == 60:
            self.last_60_ticks[self.market_id_index[market_id]] = self.last_60_ticks[self.market_id_index[market_id]][1:]
            self.last_60_ticks[self.market_id_index[market_id]].append(book_state["L1-BidPrice"])
        elif len(self.last_60_ticks) <= 60:
            self.last_60_ticks[self.market_id_index[market_id]].append(book_state["L1-BidPrice"])

        # get a prediction from the model
        # todo: consider moving this to ontime if it takes too long
        if (len(self.last_60_ticks[self.market_id_index[market_id]]) == 60 and \
            self.last_60_ticks[self.market_id_index[market_id]][0] != self.last_60_ticks[self.market_id_index[market_id]][-1]):
            
            if market_id == "Adidas":
                #print(book_state["TIMESTAMP_UTC"])
                global model_Adidas
                self.Adidas_ticks = pd.DataFrame(columns=[str(i) for i in range(0, 60)])               
                self.Adidas_ticks.loc[len(self.Adidas_ticks)] = self.last_60_ticks[self.market_id_index[market_id]]
                #self.Adidas_ticks.loc[len(self.Adidas_ticks)] = self.temp
                #prediction = self.model_Adidas.predict(self.Adidas_ticks)
                prediction = model_Adidas.predict(self.Adidas_ticks, verbose=0)
                #print(self.Adidas_ticks.head())
                #self.Adidas_ticks = self.Adidas_ticks[:-1]
                #print(self.Adidas_ticks.head())
                #self.Adidas_ticks = pd.DataFrame(columns=[str(i) for i in range(0, 60)])
                   
            elif market_id == "Allianz":
                global model_Allianz
                self.Allianz_ticks = pd.DataFrame(columns=[str(i) for i in range(0, 60)])
                self.Allianz_ticks.loc[len(self.Allianz_ticks)] = self.last_60_ticks[self.market_id_index[market_id]]
                prediction = model_Allianz.predict(self.Allianz_ticks, verbose=0)
                #self.Allianz_ticks = self.Allianz_ticks[:-1]
                
            elif market_id == "BASF":
                global model_BASF
                self.BASF_ticks = pd.DataFrame(columns=[str(i) for i in range(0, 60)])
                self.BASF_ticks.loc[len(self.BASF_ticks)] = self.last_60_ticks[self.market_id_index[market_id]]
                prediction = model_BASF.predict(self.BASF_ticks, verbose=0)
                #self.BASF_ticks = self.BASF_ticks[:-1]
                
            elif market_id == "Bayer":
                global model_Bayer
                self.Bayer_ticks = pd.DataFrame(columns=[str(i) for i in range(0, 60)])
                self.Bayer_ticks.loc[len(self.Bayer_ticks)] = self.last_60_ticks[self.market_id_index[market_id]]
                prediction = model_Bayer.predict(self.Bayer_ticks, verbose=0)
                #self.Bayer_ticks = self.Bayer_ticks[:-1]
                
            elif market_id == "BMW":
                global model_BMW
                self.BMW_ticks = pd.DataFrame(columns=[str(i) for i in range(0, 60)])
                self.BMW_ticks.loc[len(self.BMW_ticks)] = self.last_60_ticks[self.market_id_index[market_id]]
                prediction = model_BMW.predict(self.BMW_ticks, verbose=0)
                #self.BMW_ticks = self.BMW_ticks[:-1]
                
            elif market_id == "Continental":
                global model_Continental
                self.Continental_ticks = pd.DataFrame(columns=[str(i) for i in range(0, 60)])
                self.Continental_ticks.loc[len(self.Continental_ticks)] = self.last_60_ticks[self.market_id_index[market_id]]
                prediction = model_Continental.predict(self.Continental_ticks, verbose=0)
                #self.Continental_ticks = self.Continental_ticks[:-1]
                
            elif market_id == "Covestro":
                global model_Covestro
                self.Covestro_ticks = pd.DataFrame(columns=[str(i) for i in range(0, 60)])
                self.Covestro_ticks.loc[len(self.Covestro_ticks)] = self.last_60_ticks[self.market_id_index[market_id]]
                prediction = model_Covestro.predict(self.Covestro_ticks, verbose=0)
                #self.Covestro_ticks = self.Covestro_ticks[:-1]
                
            elif market_id == "Daimler":
                global model_Daimler
                self.Daimler_ticks = pd.DataFrame(columns=[str(i) for i in range(0, 60)])
                self.Daimler_ticks.loc[len(self.Daimler_ticks)] = self.last_60_ticks[self.market_id_index[market_id]]
                prediction = model_Daimler.predict(self.Daimler_ticks, verbose=0)
                #self.Daimler_ticks = self.Daimler_ticks[:-1]
                
            elif market_id == "DeutscheBank":
                global model_DeutscheBank
                self.DeutscheBank_ticks = pd.DataFrame(columns=[str(i) for i in range(0, 60)])
                self.DeutscheBank_ticks.loc[len(self.DeutscheBank_ticks)] = self.last_60_ticks[self.market_id_index[market_id]]
                prediction = model_DeutscheBank.predict(self.DeutscheBank_ticks, verbose=0)
                #self.DeutscheBank_ticks = self.DeutscheBank_ticks[:-1]
                
            elif market_id == "DeutscheBörse":
                global model_DeutscheBoerse
                self.DeutscheBoerse_ticks = pd.DataFrame(columns=[str(i) for i in range(0, 60)])
                self.DeutscheBoerse_ticks.loc[len(self.DeutscheBoerse_ticks)] = self.last_60_ticks[self.market_id_index[market_id]]
                prediction = model_DeutscheBoerse.predict(self.DeutscheBoerse_ticks, verbose=0)
                #self.DeutscheBoerse_ticks = self.DeutscheBoerse_ticks[:-1]
            
            
            if prediction[0][0] >= 0.5:
                assigned_guess[market_id] = "UP"
            elif prediction[0][0] < 0.5:
                #print(prediction)
                assigned_guess[market_id] = "DOWN"
        else:
            assigned_guess[market_id] = ""
        
        total_trades = self.market_interface.get_filtered_trades(market_id)
        total_trades_num = len(total_trades)
        
        if total_trades_num > self.total_trades[self.market_id_index[market_id]]:
            for order in self.market_interface.get_filtered_orders(market_id,
                                                                   status="ACTIVE", side="buy"):
                self.market_interface.cancel_order(order)
            #[self.market_interface.cancel_order(order) for order in self.market_interface.get_filtered_orders(market_id, status="ACTIVE", side="buy")]
                #if market_id == "BASF":
                #print("########################")
                #print("orders gecancelt")
            for order in self.market_interface.get_filtered_orders(market_id,
                                                                   status="ACTIVE", side="sell"):
                self.market_interface.cancel_order(order)
            #[self.market_interface.cancel_order(order) for order in self.market_interface.get_filtered_orders(market_id, status="ACTIVE", side="buy")]
                #if market_id == "BASF":
                #print("########################")
                #print("orders gecancelt")
        
        self.total_trades[self.market_id_index[market_id]] = total_trades_num
        #self.total_trades[self.market_id_index[market_id]] = len(total_trades)
        
        """#cancel old orders
        #if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE")) > 0:
        #    [self.market_interface.cancel_order(order) for order in
        #     self.market_interface.get_filtered_orders(market_id, status="ACTIVE")]"""

        if self.market_interface.exposure_left >= 50000:
            self.cancel_all = False
            # wait till all market orders have been hit
            if self.rebalance_orders_submitted and not self.market_interface.get_filtered_orders(status="ACTIVE"):
                """#print("######################")
                #print(self.assigned_guess)
                for item in self.market_interface.pnl_realized.items():
                    self.profit_and_loss[self.market_id_index[item[0]]].append(
                        item[1])
                print(self.profit_and_loss)
                for key in self.market_id_index.keys():
                    try:
                        pnl = self.profit_and_loss[self.market_id_index[key]][-1]
                        pnl_2 = self.profit_and_loss[self.market_id_index[key]][-2]
                        pnl_3 = self.profit_and_loss[self.market_id_index[key]][-3]
                        # if pnl < self.profit_and_loss[self.market_id_index[key]][-2] and self.assigned_guess[key] == "UP":
                        if (pnl < pnl_2) and (pnl_2 < pnl_3) and self.assigned_guess[key] == "UP":
                            self.assigned_guess[key] = "DOWN"
                        elif (pnl < pnl_2) and (pnl_2 < pnl_3) and self.assigned_guess[key] == "DOWN":
                            self.assigned_guess[key] = "UP"

                        #self.profit_and_loss[self.market_id_index[key]] = self.profit_and_loss[self.market_id_index[key]][:0]
                        self.profit_and_loss[self.market_id_index[key]] = []
                    except (KeyError, IndexError):
                        print("IndexError")"""

                self.rebalance_orders_submitted = False
                self.trading_phase = True
                #print(self.assigned_guess)
                
            #if we dont have quotable Spread, don´t sumbitt new and cancel old:
            if self.spread_market[market_id] > self.max_spread_quoted:
                ####
                trades_buy = self.market_interface.get_filtered_trades(market_id, side="buy")
                trades_sell = self.market_interface.get_filtered_trades(market_id, side="sell")
                # quantity per market
                quantity_buy = sum(t.quantity for t in trades_buy)
                quantity_sell = sum(t.quantity for t in trades_sell)
                quantity_unreal = quantity_buy - quantity_sell #>0 long, <0, short 
                #go out of the market:
                if quantity_unreal < 0:
                    self.market_interface.submit_order(market_id, "buy", quantity=abs(quantity_unreal))
                elif quantity_unreal > 0:
                    self.market_interface.submit_order(market_id, "sell", quantity=abs(quantity_unreal))

            #if we have a quotable Spread, the agent always quotes new orders:
            else:
                if self.trading_phase:
                #if True:
                    try:
                        if not self.market_interface.get_filtered_orders(market_id, status="ACTIVE"):
                            if not assigned_guess[market_id]:
                                #print(self.assigned_guess)
                                self.market_interface.submit_order(market_id, side="buy", quantity=self.min_quant_bid[market_id], limit=book_state["L2-BidPrice"])
                                self.market_interface.submit_order(market_id, side="sell", quantity=self.min_quant_ask[market_id], limit=book_state["L2-AskPrice"])
                            elif assigned_guess[market_id] == "UP":
                                #print(self.assigned_guess)
                                self.market_interface.submit_order(market_id, side="buy", quantity=math.floor(self.min_quant_bid[market_id] * 1.5), limit=book_state["L1-BidPrice"])
                                self.market_interface.submit_order(market_id, side="sell", quantity=self.min_quant_bid[market_id], limit=book_state["L2-AskPrice"])
                            elif assigned_guess[market_id] == "DOWN":
                                #print(self.assigned_guess)
                                self.market_interface.submit_order(market_id, side="buy", quantity=self.min_quant_ask[market_id], limit=book_state["L2-BidPrice"])
                                self.market_interface.submit_order(market_id, side="sell", quantity=math.floor(self.min_quant_ask[market_id] * 1.5), limit=book_state["L1-AskPrice"])
                            else:
                                pass
                    except KeyError:
                        print("Key Error")
        #less exposure than 95% left:
        else:
            if not self.cancel_all:
                # cancel all outstanding orders
                for order in self.market_interface.get_filtered_orders(status="ACTIVE", side="buy"):
                    self.market_interface.cancel_order(order)
                for order in self.market_interface.get_filtered_orders(status="ACTIVE", side="sell"):
                    self.market_interface.cancel_order(order)
                self.cancel_all = True
                #print("orders gecancelt")
            if not self.market_interface.get_filtered_orders(status="ACTIVE"):
                self.exposure_check = self.exposure_check + 1
                for market_id in self.market_interface.market_state_list.keys():
                    trades_buy = self.market_interface.get_filtered_trades(market_id, side="buy")
                    trades_sell = self.market_interface.get_filtered_trades(market_id, side="sell")
                    # quantity per market
                    quantity_buy = sum(t.quantity for t in trades_buy)
                    quantity_sell = sum(t.quantity for t in trades_sell)
                    quantity_unreal = quantity_buy - quantity_sell  # >0 long, <0, short
                    # go out of the market:
                    if quantity_unreal != 0:
                        if quantity_unreal < 0:
                            self.market_interface.submit_order(market_id, "buy", quantity=abs(quantity_unreal))
                        elif quantity_unreal > 0:
                            self.market_interface.submit_order(market_id, "sell", quantity=abs(quantity_unreal))
                self.rebalance_orders_submitted = True
                #print("rebalance_orders_subbmitted = True")
                self.trading_phase = False
        

    def on_trade(self, market_id: str, trades_state: pd.Series):

#############Evaluation################################
        ##market VWAP:
        D_V_M = sum(vol * price for vol, price in zip(trades_state.Volume, trades_state.Price))
        T_V_M = sum(trades_state.Volume)
        if market_id in self.VWAP_Market:  # to initlise VWAP market, until it has 2 value
            self.D_V_M[market_id] = self.D_V_M[market_id] + D_V_M
            self.T_V_M[market_id] = self.T_V_M[market_id] + T_V_M
        else:
            self.D_V_M[market_id] = D_V_M
            self.T_V_M[market_id] = T_V_M
        self.VWAP_Market[market_id] = self.D_V_M[market_id] / self.T_V_M[market_id]

    def on_time(self, timestamp: pd.Timestamp, timestamp_next: pd.Timestamp):
#############Evaluation################################
        ##reset iterative variables:
        if self.sess_length == datetime.timedelta(0):
            self.titm = {}  # Time in the market dictionary
            self.VWAP = {}  # stores the VWAP realized of the agent per share
            self.VWAP_Market = {}  # stores the Market VWAP realized
            self.D_V_M = {}  # accumalates dollar volume traded per share in the back testing session
            self.T_V_M = {}  # accumalates volume traded per share in the back testing session
            self.VWAP_buy = {}
            self.VWAP_sell = {}
            self.VWAP_Score = {}  # stores the VWAP Score of market VWAP and share VWAP
            self.trigger_storage_stop_loss = {}
            self.trigger_storage_take_prof = {}
            self.exposure_check = 0
            self.exposure_stor = [0]
            self.iterer = datetime.timedelta(seconds=30)
            self.milliseconds = datetime.timedelta(milliseconds=100)
            self.last_60_ticks = [[] for i in range(self.number_of_stocks)]
            
            self.last_60_ticks = [[] for i in range(self.number_of_stocks)]
            self.total_trades = [0 for i in range(self.number_of_stocks)]
            self.cancel_all = False
            self.trading_phase = False
            self.rebalance_orders_submitted = False
            self.assigned_guess = {}
            
        if not self.assigned_guess:
            self.assigned_guess = {key: "" for key in self.market_interface.market_state_list.keys()}
            self.trading_phase = True

        # calculate the Session length:
        delta = timestamp_next - timestamp
        self.sess_length = self.sess_length + delta
        #store every Second:
        if self.milliseconds < self.sess_length:
            self.milliseconds = self.milliseconds + datetime.timedelta(milliseconds=100)
             ###Calculate Time in the market per Share
            for market_id in self.market_interface.market_state_list.keys():           
                #if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE", side= "sell" )) and \
                #    len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE", side= "buy" )) != 0:
                if self.market_interface.get_filtered_orders(market_id, status="ACTIVE", side= "sell" ) and \
                    self.market_interface.get_filtered_orders(market_id, status="ACTIVE", side= "buy" ):                    
                #if len(self.market_interface.get_filtered_orders(market_id, status="ACTIVE")): 
                    if market_id not in self.titm.keys():
                        self.titm[market_id] = datetime.timedelta(0)
                    else:
                        self.titm[market_id] = self.titm[market_id] + datetime.timedelta(milliseconds=100)

        #store exposure development
        if self.milliseconds >= self.iterer:
            exposure = self.market_interface.exposure
            total_net_exposure = sum(exposure.values())
            self.exposure_stor.append(total_net_exposure)
            self.iterer = self.iterer + datetime.timedelta(seconds = 30)

        ######store values at the end of a session###
        if timestamp == timestamp_next:
            """for market_id in self.market_interface.market_state_list.keys():
                [print(order.__str__()) for order in self.market_interface.get_filtered_orders(market_id)]"""
            
            trades = self.market_interface.get_filtered_trades()
            quantity = sum(t.quantity for t in trades)
            if quantity > 0:
                self.VWAP["Total"] = sum(t.quantity * t.price for t in trades) / quantity
            else:
                self.VWAP["Total"] = 0

            for market_id in self.market_interface.market_state_list.keys():
                ###current VWAP of agent all together ###

                ###current VWAP of agent per share ###
                trades = self.market_interface.get_filtered_trades(market_id)
                quantity = sum(t.quantity for t in trades)
                if quantity > 0:
                    self.VWAP[market_id] = sum(t.quantity * t.price for t in trades) / quantity
                else:
                    self.VWAP[market_id] = 0

                ###current sell VWAP of agent per share ###
                trades_sell = self.market_interface.get_filtered_trades(market_id, side="sell")
                quantity_sell = sum(t.quantity for t in trades_sell)
                if quantity_sell > 0:
                    self.VWAP_sell[market_id] = sum(t.quantity * t.price for t in trades_sell) / quantity_sell
                else:
                    self.VWAP_sell[market_id] = 0

                ###current buy VWAP of agent per share ###
                trades_buy = self.market_interface.get_filtered_trades(market_id, side="buy")
                quantity_buy = sum(t.quantity for t in trades_buy)
                if quantity_buy > 0:
                    self.VWAP_buy[market_id] = sum(t.quantity * t.price for t in trades_buy) / quantity_buy
                else:
                    self.VWAP_buy[market_id] = 0

                ### Calculate VWAP_Score
                try:
                    self.VWAP_Score[market_id] = quantity_buy * (self.VWAP_Market[market_id] - self.VWAP_buy[market_id]) + \
                                                 quantity_sell * (self.VWAP_sell[market_id] - self.VWAP_Market[market_id])
                except KeyError:
                    pass


            titm_df = pd.DataFrame.from_dict(self.titm, orient='index')
            proz_titm_df = titm_df / self.sess_length
            VWAP_score_df = pd.DataFrame.from_dict(self.VWAP_Score, orient='index')
            pnl_df = pd.DataFrame.from_dict(self.market_interface.pnl_realized, orient='index')
            pnl_unr_df = pd.DataFrame.from_dict(self.market_interface.pnl_unrealized, orient='index')
            trades_df = pd.DataFrame()
            volume = {}
            trading_costs = {}
            num_shares = {}
            num_shares_b = {}
            num_shares_s = {}
            num_quotes = {}

            #get traded volume, costs and n of stop loss and take prof:
            for market_id in self.market_interface.market_state_list.keys():
                trades = self.market_interface.get_filtered_trades(market_id)
                trades_b = self.market_interface.get_filtered_trades(market_id, side="buy")
                trades_s = self.market_interface.get_filtered_trades(market_id, side="sell")
                shares_total = sum(t.quantity for t in trades)
                shares_b = sum(t.quantity for t in trades_b)
                shares_s = sum(t.quantity for t in trades_s)
                quotes = self.market_interface.get_filtered_orders(market_id)
                quotes = len(quotes) / 2
                quantity = sum(t.quantity * t.price for t in trades)
                volume[market_id] = quantity
                trading_costs[market_id] = quantity * self.market_interface.transaction_cost_factor

                trades_df.loc[market_id,0] = len(self.market_interface.get_filtered_trades(market_id))
                num_shares[market_id] = shares_total
                num_shares_b[market_id] = shares_b
                num_shares_s[market_id] = shares_s
                num_quotes[market_id] = quotes

                if market_id not in self.trigger_storage_take_prof:
                    self.trigger_storage_take_prof[market_id] = 0
                if market_id not in self.trigger_storage_stop_loss:
                    self.trigger_storage_stop_loss[market_id] = 0

            trigger_storage_take_prof_df = pd.DataFrame.from_dict(self.trigger_storage_take_prof, orient='index')
            trigger_storage_stop_loss_df = pd.DataFrame.from_dict(self.trigger_storage_stop_loss, orient='index')
            volume_df = pd.DataFrame.from_dict(volume, orient='index')
            trading_costs_df = pd.DataFrame.from_dict(trading_costs, orient='index')
            num_shares_df = pd.DataFrame.from_dict(num_shares, orient="index")
            num_shares_b_df = pd.DataFrame.from_dict(num_shares_b, orient="index")
            num_shares_s_df = pd.DataFrame.from_dict(num_shares_s, orient="index")
            num_quotes_df = pd.DataFrame.from_dict(num_quotes, orient="index")
            """try:
                titm_df.columns = [str(timestamp)]
                proz_titm_df.columns = ["%-Time in the Market"]
                VWAP_score_df.columns = ["VWAP_Score"]
                pnl_df.columns = ["pnl_realized"]
                pnl_unr_df.columns = ["pnl_unrealized"]
                trades_df.columns = ["n_trades"]
                trigger_storage_take_prof_df.columns = ["n_take_prof"]
                trigger_storage_stop_loss_df.columns = ["n_stopp_loss"]
                volume_df.columns = ["dollar_volume_traded"]
                trading_costs_df.columns = ["trading_costs"]
            except ValueError:
                pass"""
            titm_df.columns = [str(timestamp)]
            proz_titm_df.columns = ["%-Time in the Market"]
            VWAP_score_df.columns = ["VWAP_Score"]
            pnl_df.columns = ["pnl_realized"]
            pnl_unr_df.columns = ["pnl_unrealized"]
            trades_df.columns = ["n_trades"]
            trigger_storage_take_prof_df.columns = ["n_take_prof"]
            trigger_storage_stop_loss_df.columns = ["n_stopp_loss"]
            volume_df.columns = ["dollar_volume_traded"]
            trading_costs_df.columns = ["trading_costs"]
            num_shares_df.columns = ["number of shares"]
            num_shares_b_df.columns = ["shares bought"]
            num_shares_s_df.columns = ["shares sold"]
            num_quotes_df.columns = ["number of quotes"]


            titm_df = titm_df.transpose()
            proz_titm_df = proz_titm_df.transpose()
            VWAP_score_df = VWAP_score_df.transpose()
            pnl_df = pnl_df.transpose()
            pnl_unr_df = pnl_unr_df.transpose()
            trades_df = trades_df.transpose()
            trigger_storage_take_prof_df = trigger_storage_take_prof_df.transpose()
            trigger_storage_stop_loss_df = trigger_storage_stop_loss_df.transpose()
            volume_df = volume_df.transpose()
            trading_costs_df = trading_costs_df.transpose()
            num_shares_df = num_shares_df.transpose()
            num_shares_b_df = num_shares_b_df.transpose()
            num_shares_s_df = num_shares_s_df.transpose()
            num_quotes_df = num_quotes_df.transpose()

            global result_exposure, result_per_Share, result_matrix
            result_exposure = result_exposure.append(self.exposure_stor)

            result_per_Share = result_per_Share.append([titm_df,proz_titm_df,VWAP_score_df,pnl_df,trades_df,trigger_storage_take_prof_df,
                                                        trigger_storage_stop_loss_df,volume_df,trading_costs_df,pnl_unr_df, num_shares_df, num_shares_b_df, num_shares_s_df, num_quotes_df])

            result_matrix = result_matrix.append(
                {
                    "timestamp": str(timestamp),
                    "exposure": self.market_interface.exposure_total,
                    'pnl': self.market_interface.pnl_realized_total,
                    "n_trades": len(self.market_interface.get_filtered_trades()),
                    "costs": self.market_interface.transaction_cost,
                    "n_orders": len(self.market_interface.get_filtered_orders()),
                    "session_length": self.sess_length,
                    "VWAP_Total_Agent": self.VWAP["Total"],
                    "Exposure_left < 0": self.exposure_check,
                    "pnl_unrealized": self.market_interface.pnl_unrealized_total,
                }
                , ignore_index=True)

            self.last_60_ticks = [[] for i in range(self.number_of_stocks)]
            self.assigned_guess = {}
            self.assigned_guess = {key: "" for key in self.market_interface.market_state_list.keys()}
            self.total_trades = [0 for i in range(self.number_of_stocks)]
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
            #self.trading_phase = False
            self.profit_and_loss = [[] for i in range(self.number_of_stocks)]
            self.cancel_all = False
                          
            
if __name__ == "__main__":


    identifier_list = [
       # ADIDAS
       "Adidas.BOOK", "Adidas.TRADES",
       # ALLIANZ
       "Allianz.BOOK", "Allianz.TRADES",
       # BASF
       "BASF.BOOK", "BASF.TRADES",
       # Bayer
       "Bayer.BOOK", "Bayer.TRADES",
       # BMW
       "BMW.BOOK", "BMW.TRADES",
       # Continental
       "Continental.BOOK", "Continental.TRADES",
       # Covestro
       "Covestro.BOOK", "Covestro.TRADES",
       # Daimler
       "Daimler.BOOK", "Daimler.TRADES",
       # Deutsche Bank
       "DeutscheBank.BOOK", "DeutscheBank.TRADES",
       # DeutscheBörse
       "DeutscheBörse.BOOK", "DeutscheBörse.TRADES",
    ]

    agent = SimpleAgent(
        name="Bench_Agent_naiv",
        number_of_stocks=(len(identifier_list) // 2),
    )

    backtest = Backtest(
        agent=agent,
    )
    backtest.run_episode_generator(identifier_list=identifier_list,
                                   date_start="2021-02-01",
                                   date_end="2021-02-28",
                                   episode_interval=60,
                                   episode_shuffle=True,
                                   episode_buffer=15,
                                   episode_length=75,
                                   num_episodes=10,
                                   seed=5,
    )




#print results in excel:
result_per_Share.iloc[:1] = result_per_Share.iloc[:1].astype(str)
result_matrix["session_length"] = result_matrix["session_length"].astype(str)
"""name_of_file = "result_" + agent.name + ".xlsx"
writer = pd.ExcelWriter(name_of_file)
result_matrix.to_excel(writer,"result_matrix")
result_per_Share.to_excel(writer,"result_per_Share")
result_exposure.to_excel(writer, "exposure_history")
writer.save()"""

result_per_Share.to_csv("./log_regression/results_per_share_session345678910.csv")
result_matrix.to_csv("./log_regression/result_matrix_session345678910.csv")
result_exposure.to_csv("./log_regression/result_exposure_session345678910.csv")

#result_per_Share.to_csv("./log_regression/number_of_shares2.csv")