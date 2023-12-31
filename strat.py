from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import talib, numpy
import pandas as pd
#todo: learn divergence, know more about the rules of the contest

class AlgoEvent:
    def __init__(self):
        
        self.lasttradetime = datetime(2000,1,1)
        self.start_time = None # the starting time of the trading
        self.ma_len = 20 # len of arrays of Moving Average
        self.rsi_len = 14 # len of window size in rsi calculation
        self.wait_time = self.ma_len # in days
        self.arr_close_dict = {} # key to the corresponding arr_close
        self.inst_data = {} # for storing data of the instruments
        self.general_period = 14 # general period for indicator 
        self.bb_sdwidth = 2
        self.fastperiod = 5 
        self.midperiod = 8
        self.slowperiod = 14
        self.longperiod = 50
        self.squeezeThreshold_percentile = 0.2
        self.risk_reward_ratio = 2.5 # take profit level : risk_reward_ratio * stoploss
        self.stoploss_atrlen = 2.5 # width of atr for stoplsos
        self.allocationratio_per_trade = 0.3

        self.risk_limit_portfolio = 0.2
        self.cooldown = 10
        self.openOrder = {} # existing open position for updating stoploss and checking direction
        self.netOrder = {} # existing net order
        
        self.sorted_score1_list = [] # sorted list of (key, score1) in decreasing order
        self.sorted_score2_3_list = [] # sorted list of (key, score2_3) in decreasing order
        
        self.temp_traded_dict = {"ZeroDay": [], "OneDay": [], "TwoDay": []} 

    def start(self, mEvt):
        self.myinstrument = mEvt['subscribeList'][0]
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.update_portfolio_sl(sl=self.risk_limit_portfolio, resume_after=60*60*24*self.cooldown)
        self.evt.start()


    def on_bulkdatafeed(self, isSync, bd, ab):
        # set start time and inst_data in bd on the first call of this function
       
        if not self.start_time:
            self.evt.consoleLog(f"start")
            
            
            self.start_time = bd[self.myinstrument]['timestamp']
            for key in bd:
                self.inst_data[key] = {
                    "arr_close": numpy.array([]),
                    "high_price": numpy.array([]),
                    "low_price": numpy.array([]),
                    'arr_fastMA': numpy.array([]),
                    'arr_midMA': numpy.array([]),
                    'arr_slowMA': numpy.array([]),
                    'upper_bband': numpy.array([]),
                    'lower_bband': numpy.array([]),
                    'BB_width': numpy.array([]),
                    'atr': numpy.array([]),
                    'K': numpy.array([]), # Stoch rsi K
                    'D': numpy.array([]), # Stoch rsi D
                    'entry_signal': 0,
                    
                    'score1': 0, # higher better
                    'score2_3': 0 # higher better
                }
            self.no_of_inst = len(bd.keys())
                
                
        # check if it is decision time
        if bd[self.myinstrument]['timestamp'] >= self.lasttradetime + timedelta(hours=24):
            # update inst_data's arr close, highprice and lowprice, and MA lines
            self.lasttradetime = bd[self.myinstrument]['timestamp']
            
            for key in bd:
                inst_data = self.inst_data[key]
                
                # Collecting price data
                inst_data['high_price'] = numpy.append(inst_data['high_price'], bd[key]['highPrice'])
                inst_data['arr_close'] = numpy.append(inst_data['arr_close'], bd[key]['lastPrice'])
                inst_data['low_price'] = numpy.append(inst_data['low_price'], bd[key]['lowPrice'])
                
                time_period = self.ma_len * 2
                
                # keep the most recent observations
                inst_data['high_price'] = inst_data['high_price'][-time_period::]
                inst_data['arr_close'] = inst_data['arr_close'][-time_period::]
                inst_data['low_price'] = inst_data['low_price'][-time_period::]
                
                sma = self.find_sma(inst_data['arr_close'], self.ma_len)
                sd = numpy.std(inst_data['arr_close'])
                inst_data['upper_bband'] = numpy.append(inst_data['upper_bband'], sma + self.bb_sdwidth*sd)
                inst_data['lower_bband'] = numpy.append(inst_data['lower_bband'], sma - self.bb_sdwidth*sd)
                inst_data['BB_width'] = inst_data['upper_bband'] - inst_data['lower_bband']
                # Calculating indicator value
                inst_data['atr'] = talib.ATR(inst_data['high_price'], inst_data['low_price'], inst_data['arr_close'], timeperiod = self.general_period)
                
                
                inst_data['arr_fastMA'] = talib.EMA(inst_data['arr_close'], self.fastperiod)
                inst_data['arr_midMA'] = talib.EMA(inst_data['arr_close'], self.midperiod)
                inst_data['arr_slowMA'] = talib.EMA(inst_data['arr_close'], self.slowperiod)
                inst_data['arr_longMA'] = talib.EMA(inst_data['arr_close'], self.longperiod)
                K, D = self.stoch_rsi(inst_data['arr_close'], k = 3, d = 3, period = 14)
                inst_data['K'], inst_data['D'] = numpy.append(inst_data['K'], K), numpy.append(inst_data['D'], D)
                
                
                inst_data['entry_signal'] = self.get_entry_signal(inst_data)
                
                #self.evt.consoleLog(f"entry singal: {inst_data['entry_signal']}")
                
                stoploss = inst_data['atr'][-1] * self.stoploss_atrlen
                if self.openOrder:
                    self.update_stoploss(key, stoploss)
                
                
                # test
                #self.evt.consoleLog(f"high price (len of {len(inst_data['high_price'])}): {inst_data['high_price']}")
                #self.evt.consoleLog(f"arr_close: {inst_data['arr_close']}")
                #self.evt.consoleLog(f"low_price: {inst_data['low_price']}")
                
                #self.evt.consoleLog(f"upper_bband: {inst_data['upper_bband']}")
                #self.evt.consoleLog(f"lower_bband: {inst_data['lower_bband']}")
                #self.evt.consoleLog(f"BB_width: {inst_data['BB_width']}")
                
                #self.evt.consoleLog(f"atr: {inst_data['atr']}")
                
                #self.evt.consoleLog(f"arr_fastMA: {inst_data['arr_fastMA']}")
                #self.evt.consoleLog(f"arr_midMA: {inst_data['arr_midMA']}")
                #self.evt.consoleLog(f"arr_slowMA: {inst_data['arr_slowMA']}")
                
                #self.evt.consoleLog(f"K: {inst_data['K']}")
                #self.evt.consoleLog(f"D: {inst_data['D']}")
                
            # ranking for signal 2 and 3 based on BBW (favours less BBW)
            # get scores for ranking
            #self.get_score2_3(bd, self.inst_data)
            self.get_scores(bd, self.inst_data)
            self.get_sorted_score_lists(bd, self.inst_data)
            
            # update temp_traded_dict
            self.temp_traded_dict["OneDay"] = self.temp_traded_dict["ZeroDay"]
            self.temp_traded_dict["TwoDay"] = self.temp_traded_dict["OneDay"]
            self.temp_traded_dict["ZeroDay"] = []
            
            trade_2_3 = 0
            trade_1 = 0
            
            # execute trading strat based on score2_3 (non-ranging market)
            for (key, score2_3) in self.sorted_score2_3_list:
                if trade_2_3 >= 2:
                    break
                # check if recently traded
                if key in self.temp_traded_dict["OneDay"] or key in self.temp_traded_dict["TwoDay"] or key in self.temp_traded_dict["ZeroDay"]:
                    return
                
                if self.inst_data[key]['entry_signal'] in [-3,-2,2,3]:
                    self.execute_strat(bd, key)
                    self.temp_traded_dict["ZeroDay"].append(key)
                    trade_2_3 += 1
                     # only trade once
                    
            # execute the trading strat based on score1 (ranging market), but exclude those that excuted b4
            for (key, score1) in self.sorted_score1_list:
                if trade_1 >= 2:
                    break
                # check if recently traded
                if key in self.temp_traded_dict["OneDay"] or key in self.temp_traded_dict["TwoDay"] or key in self.temp_traded_dict["ZeroDay"]:
                    return
                
                if self.inst_data[key]['entry_signal'] in [-1,1]:
                    self.execute_strat(bd, key)
                    self.temp_traded_dict["ZeroDay"].append(key)
                    trade_1 += 1
            
            self.no_of_trade_today = max(trade_1+ trade_2_3, 1)
                
            
            
    def on_marketdatafeed(self, md, ab):
        pass

    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        self.openOrder = oo
        self.netOrder = op
    
    
    def find_sma(self, data, window_size):
        return data[-window_size::].sum()/window_size
    
    def momentumFilter(self, APO, MACD, RSIFast, RSIGeneral, AROONOsc, strict):
        
        
        # APO rising check
        APORising = False
        if numpy.isnan(APO[-1]) or numpy.isnan(APO[-2]):
            APORising = False
        elif int(APO[-1]) > int(APO[-2]):
            APORising = True
        
        # macd rising check
        MACDRising = False
        if numpy.isnan(MACD[-1]) or numpy.isnan(MACD[-2]):
            MACDRising = False
        elif int(MACD[-1]) > int(MACD[-2]):
            MACDRising = True
        
        # RSI check (additional)
        RSIFastRising, RSIGeneralRising = False, False
        if numpy.isnan(RSIFast[-1]) or numpy.isnan(RSIFast[-2]) or numpy.isnan(RSIGeneral[-2]) or numpy.isnan(RSIGeneral[-2]):
            RSIFastRising, RSIGeneralRising = False, False
        else:
            if int(RSIFast[-1]) > int(RSIFast[-2]):
                RSIFastRising = True
            if int(RSIGeneral[-1]) > int(RSIGeneral[-2]):
                RSIGeneralRising = True
            
        # aroonosc rising check
        AROON_direction = 0 # not moving
        if numpy.isnan(AROONOsc[-1]) or numpy.isnan(AROONOsc[-2]):
            AROON_direction = 0
        elif int(AROONOsc[-1]) > int(AROONOsc[-2]):
            AROON_direction = 1 # moving upwawrds
        elif int(AROONOsc[-1]) < int(AROONOsc[-2]):
            AROON_direction = -1 # moving downwards
        else:
            AROON_direction = 0 # not moving

        # aroon oscillator positive check
        AROON_positive = False
        if numpy.isnan(AROONOsc[-1]):
            AROON_positive = False
        elif int(AROONOsc[-1]) > 0:
            AROON_positive = True
        
        if strict:
            if (APO[-1] > 0) and (RSIFast[-1] > 50 or RSIFastRising or RSIGeneralRising) and (MACDRising or AROON_direction == 1 or AROON_positive):
                return 1 # Bullish 
                
            elif (APO[-1] < 0) and (RSIFast[-1] < 50 or not RSIFastRising or not RSIGeneralRising) and (not MACDRising or AROON_direction == -1 or not AROON_positive):
                return -1 # Bearish
            else:
                return 0 # Neutral
                
        else:
            if (APO[-1] > 0) or (RSIFast[-1] > 50 or RSIFastRising or RSIGeneralRising) or (MACDRising or AROON_direction == 1 or AROON_positive):
                return 1 # Bullish 
                
            elif (APO[-1] < 0) or (RSIFast[-1] < 50 or not RSIFastRising or not RSIGeneralRising) or (not MACDRising or AROON_direction == -1 or not AROON_positive):
                return -1 # Bearish
            else:
                return 0 # Neutral
            
    def testrangingFilter(self, ADXR, AROONOsc, MA_same_direction, rsi): 
        score = 0
        score += (100-ADXR[-1])/100
        score *= (100 - abs(AROONOsc[-1]))/100
        return score >= 0.3
        
    def rangingFilter(self, ADXR, AROONOsc, MA_same_direction, rsi, stream):
        if stream == 2 or stream == 3:
            if (ADXR[-1] < 30) or abs(AROONOsc[-1]) < 50 or not MA_same_direction:
                return True # ranging market
            else:
                return False
        if stream == 1:
            if (ADXR[-1] < 30) and abs(AROONOsc[-1]) < 50 and not MA_same_direction:
                return True # ranging market
            else:
                return False
            
    
    
    # get score1 (ranging market) and score2_3 for all instruments
    def get_scores(self, bd, inst_data):
        # loop once to get the min, max bbw and atr among all instrument
        min_atr, min_bbw = 1000000000, 1000000000
        max_atr, max_bbw = 0, 0
        for key in bd:
            min_bbw = min(min_bbw, inst_data[key]["BB_width"][-1])
            min_atr = min(min_atr, inst_data[key]["atr"][-1])
            
            max_bbw = max(max_bbw, inst_data[key]["BB_width"][-1])
            max_atr = max(max_atr, inst_data[key]["atr"][-1])
        
        # assign score1, and score2_3 for each instruments
        for key in bd:
            # score2_3
            inst_data[key]["score2_3"] = (max_bbw - inst_data[key]["BB_width"][-1])/ (max_bbw-min_bbw) # in [0,1]
            # score1 (bbw part) 
            inst_data[key]['score1'] = (inst_data[key]["BB_width"][-1] - min_bbw) / (max_bbw-min_bbw) # in [0,1]
            # score1 (atr part)
            if not numpy.isnan(inst_data[key]["atr"][-1]):
                inst_data[key]["score1"] += (max_atr - inst_data[key]["atr"][-1])/ (max_atr-min_atr) # in [0,2]
            # score1 normalize 
            inst_data[key]['score1'] /= 2 # in [0,1]
            
            #debug
            #self.evt.consoleLog(f"score1 {inst_data[key]['score1']}")
            #self.evt.consoleLog(f"score2_3 {inst_data[key]['score2_3']}")
    
    
    def get_sorted_score_lists(self, bd, inst_data):
        # sorting for self.sorted_score2_3_list
        sorted_list = []
        for key in bd:
            score2_3 = self.inst_data[key]["score2_3"]
            if numpy.isnan(score2_3):
                continue
            sorted_list.append((key, score2_3))
        sorted_list = sorted(sorted_list, key=lambda tup: tup[1])
        sorted_list.reverse()
        #self.evt.consoleLog(f"sorted list2_3: {sorted_list}") # debug
        
        # cut down the len
        #new_len = min((len(sorted_list)-1 ) // 2, 3)
        #sorted_list = sorted_list[0:new_len:]
        #self.evt.consoleLog(f"sorted list2_3: {sorted_list}") 
        
        self.sorted_score2_3_list = sorted_list
        
        # sorting for self.sorted_score1_list
        sorted_list = []
        for key in bd:
            score1 = self.inst_data[key]["score1"]
            if numpy.isnan(score1):
                continue
            sorted_list.append((key, score1))
        sorted_list = sorted(sorted_list, key=lambda tup: tup[1])
        sorted_list.reverse()
        #self.evt.consoleLog(f"sorted list1: {sorted_list}") # debug
        
        # cut down the len
        new_len = min((len(sorted_list)-1 ) // 2, 3)
        sorted_list = sorted_list[0:new_len:]
        #self.evt.consoleLog(f"sorted list1: {sorted_list}") 
        
        self.sorted_score1_list = sorted_list
        
        
    def get_entry_signal(self, inst_data):
        inst = inst_data
        arr_close = inst['arr_close']
        sma = self.find_sma(inst_data['arr_close'], self.ma_len)
        upper_bband, lower_bband = inst['upper_bband'][-1], inst['lower_bband'][-1]
        
        lastprice = arr_close[-1]
        # squeeze entry signal
        bbw = inst['BB_width']
        curbbw = bbw[-1]
        bb_squeeze_percentile = (sorted(bbw).index(curbbw) + 1) / len(bbw)
        squeeze = bb_squeeze_percentile < self.squeezeThreshold_percentile
        squeeze_breakout = squeeze and lastprice > upper_bband
        squeeze_breakdown = squeeze and lastprice < upper_bband
        
        
        # Use Short term MA same direction for ranging filters
        fast, mid, slow, long = inst['arr_fastMA'], inst['arr_midMA'], inst['arr_slowMA'], inst['arr_longMA']
        all_MA_up, all_MA_down, MA_same_direction = False, False, False
        if len(fast) > 1 and len(mid) > 1 and len(slow) > 1:
            all_MA_up = fast[-1] > fast[-2] and mid[-1] > mid[-2] and slow[-1] > slow[-2]
            all_MA_down = fast[-1] < fast[-2] and mid[-1] < mid[-2] and slow[-1] < slow[-2]
            MA_same_direction = all_MA_up or all_MA_down
            
        # ranging filter (to confirm moving sideway)
        adxr = talib.ADXR(inst['high_price'], inst['low_price'], inst['arr_close'], 
            timeperiod=self.general_period-1)
            
        apo = talib.APO(inst['arr_close'], self.midperiod, self.slowperiod)
        macd, signal, hist = talib.MACD(inst['arr_close'], self.fastperiod, self.slowperiod, self.midperiod)
        rsiFast, rsiGeneral = talib.RSI(inst['arr_close'], self.fastperiod), talib.RSI(inst['arr_close'], self.general_period)       
        # Calculate Aroon values
        aroon_up, aroon_down = talib.AROON(inst['high_price'], inst['low_price'], timeperiod=self.general_period)
        aroonosc = aroon_up - aroon_down
        
        #self.evt.consoleLog(f"adxr {adxr}") #adxr is an array of all nan, bug
        #self.evt.consoleLog(f"apo {apo}") 
        #self.evt.consoleLog(f"macd {macd}") 
        #self.evt.consoleLog(f"signal {signal}") 
        #self.evt.consoleLog(f"hist {hist}") 
        #self.evt.consoleLog(f"aroon_up {aroon_up}") 
        #self.evt.consoleLog(f"aroon_down {aroon_down}") 
        
        
        # Entry signal 2: stoch RSI crossover
        
        # Long Entry: K crossover D from below
        long_stoch_rsi = inst['K'][-1] > inst['D'][-1] and inst['K'][-2] < inst['D'][-1] and mid[-1] > slow[-1] and slow[-1] > long[-1]
        # Short Entry: K crossover D from above
        short_stoch_rsi = inst['K'][-1] < inst['D'][-1] and inst['K'][-2] > inst['D'][-2]  and mid[-1] < slow[-1] and slow[-1] < long[-1]
        
        

        # TODO:  classify the different type of entry signal and set take profit/ stop loss accordingly
        
        ranging1 = self.rangingFilter(adxr, aroonosc, MA_same_direction, rsiGeneral, 1)
        ranging2_3 = self.rangingFilter(adxr, aroonosc, MA_same_direction, rsiGeneral, 2)
        
        bullish1 = self.momentumFilter(apo, macd, rsiFast, rsiGeneral, aroonosc, False)
        bullish2_3 = self.momentumFilter(apo, macd, rsiFast, rsiGeneral, aroonosc, True)
        
        # check for buy
        if lastprice >= upper_bband and rsiGeneral[-1] > 70 and ranging1 and bullish1 == -1:
                return -1
        if squeeze_breakdown and not ranging2_3 and bullish2_3 == -1:
            return -2
        if short_stoch_rsi and not ranging2_3 and bullish2_3 == -1:
            return -3 
        
        #check for sell
        if lastprice <= lower_bband and rsiGeneral[-1] < 30 and ranging1 and bullish1 == 1:
            #self.evt.consoleLog("bb + rsi strat buy signal")
            return 1
        if squeeze_breakout and not ranging2_3 and bullish2_3 == 1:
            return 2
        if long_stoch_rsi and not ranging2_3 and bullish2_3 == 1:
            return 3
            
        # no signal
        return 0 
     
        
    # execute the trading strat for one instructment given the key and bd       
    def execute_strat(self, bd, key):
        
        inst = self.inst_data[key]
        lastprice =  inst['arr_close'][-1]
        position_size = self.allocate_capital( self.calculate_net_returns(inst['arr_close']), key )
        # set direction, ie decide if buy or sell, based on entry signal
        direction = 1
        if inst['entry_signal'] > 0:
            direction = 1 #long
        elif inst['entry_signal'] < 0:
            direction = -1 #short
        
        
        atr =  inst['atr'][-1]
        stoploss = self.stoploss_atrlen * atr
        takeprofit = None
        if inst['entry_signal'] == 1 or -1:
            takeprofit = (inst['upper_bband'][-1] + inst['lower_bband'][-1])/2 # use the middle band as take profit
        elif inst['entry_signal'] == 2 or 3 or -2 or -3:
            takeprofit = self.risk_reward_ratio * stoploss
        
        if key in self.openOrder and self.openOrder[key][buysell] != direction and self.openOrder[instrument]['orderRef'] == abs(inst['entry_signal']):
            # if current position exist in open order as well as opposite direction and same trading signal, close the order
            self.closeAllOrder(instrument, self.openOrder[instrument][orderRef])
            
        self.test_sendOrder(lastprice, direction, 'open', stoploss, takeprofit, position_size, key, inst['entry_signal'] )
        
    def calculate_net_returns(self, prices):
        net_returns = 0
        for i in range(1, len(prices)):
            daily_return = prices[i] - prices[i-1]
            net_returns += daily_return
        return net_returns   
        
    def allocate_capital(self, strategy_returns, key):
        inst = self.inst_data[key]
        initial_price = inst['arr_close'][0]
        if strategy_returns <= 0:
            return 0.03
        # Calculate the available capital for trading
        res = self.evt.getAccountBalance()
        bal = res["availableBalance"]
        if bal < 0.2 :
            bal = abs(bal)*10
        position_size = bal * (strategy_returns / initial_price)   
        return position_size


    def test_sendOrder(self, lastprice, buysell, openclose, stoploss, takeprofit, volume, instrument, orderRef):
        order = AlgoAPIUtil.OrderObject()
        order.instrument = instrument
        order.orderRef = orderRef
        if buysell==1:
            order.takeProfitLevel = lastprice + takeprofit
            order.stopLossLevel = lastprice - stoploss 
        elif buysell==-1:
            order.takeProfitLevel = lastprice - takeprofit
            order.stopLossLevel = lastprice + stoploss
        order.volume = volume
        order.openclose = openclose
        order.buysell = buysell
        order.ordertype = 0 #0=market_order, 1=limit_order, 2=stop_order
        self.evt.sendOrder(order)
    
    
    # Finder of Stochastic RSI
    def stoch_rsi(self, arr_close, k, d, period):
        rsi = talib.RSI(arr_close, period)
        df = pd.DataFrame(rsi)
        stochastic_rsi = 100 * (df - df.rolling(period).min()) / (df.rolling(period).max() - df.rolling(period).min())
        K = stochastic_rsi.rolling(k).mean()
        D = K.rolling(d).mean().iloc[-1].iloc[0]
        K = K.iloc[-1].iloc[0]
        return K, D 
        # K and D are returned as a value
    
    def closeAllOrder(self, instrument, orderRef):
        if not self.openOrder:
            return False
        for ID in self.openOrder:
            if self.openOrder[ID]['instrument'] == instrument and self.openOrder[ID]['orderRef'] == orderRef:
                order = AlgoAPIUtil.OrderObject(
                    tradeID = ID,
                    openclose = 'close',
                )
                self.evt.sendOrder(order)
        return True
        
        
    # ATR trailing stop implementation
    def update_stoploss(self, instrument, new_stoploss):
        for ID in self.openOrder:
            openPosition = self.openOrder[ID]
            if openPosition['instrument'] == instrument:
                lastprice = self.inst_data[instrument]['arr_close'][-1]
                if openPosition['buysell'] == 1 and openPosition['stopLossLevel'] < lastprice - new_stoploss: 
                    # for buy ordder, update stop loss if current ATR stop is higher than previous 
                    newsl_level = lastprice - new_stoploss
                    res = self.evt.update_opened_order(tradeID=ID, sl = newsl_level)
                    # update the update stop loss using ATR stop
                elif openPosition['buysell'] == -1 and lastprice + new_stoploss < openPosition['stopLossLevel']: 
                    # for buy ordder, update stop loss if current ATR stop is higher than previous 
                    newsl_level = lastprice + new_stoploss
                    res = self.evt.update_opened_order(tradeID=ID, sl = newsl_level)
                    # update the update stop loss using ATR stop
                    
        

    # utility function to find volume based on available balance
    def find_positionSize(self, lastprice):
        res = self.evt.getAccountBalance()
        availableBalance = res["availableBalance"]
        ratio = self.allocationratio_per_trade
        volume = (availableBalance*ratio) / lastprice
        total =  volume *  lastprice
        while total < self.allocationratio_per_trade * availableBalance:
            ratio *= 1.05
            volume = (availableBalance*ratio) / lastprice
            total =  volume *  lastprice
        while total > availableBalance:
            ratio *= 0.95
            volume = (availableBalance*ratio) / lastprice
            total =  volume *  lastprice
        return volume
