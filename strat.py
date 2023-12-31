
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
        self.slowperiod = 13
        self.squeezeThreshold_percentile = 0.2
        self.risk_reward_ratio = 2.5 # take profit level : risk_reward_ratio * stoploss
        self.stoploss_atrlen = 2.5 # width of atr for stoplsos
        self.allocationratio_per_trade = 0.3
        
        self.openOrder = {} # existing open position for updating stoploss and checking direction
        self.netOrder = {} # existing net order

    def start(self, mEvt):
        self.myinstrument = mEvt['subscribeList'][0]
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()


    def on_bulkdatafeed(self, isSync, bd, ab):
        # set start time and inst_data in bd on the first call of this function
        if not isSync:
            return
        if not self.start_time:
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
                }
                
                
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
                K, D = self.stoch_rsi(inst_data['arr_close'], k = 3, d = 3, period = 14)
                inst_data['K'], inst_data['D'] = numpy.append(inst_data['K'], K), numpy.append(inst_data['D'], D)
                
                
                inst_data['entry_signal'] = self.get_entry_signal(inst_data)
                stoploss = inst_data['atr'][-1] * self.stoploss_atrlen
                if key in self.openOrder:
                    self.update_stoploss(key, stoploss)
             
            
            # execute the trading strat for all instruments
            for key in bd:
                if self.inst_data[key]['entry_signal'] != 0:
                    self.execute_strat(bd, key)
            
            
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
    
    def momentumFilter(self, APO, MACD, RSIFast, RSIGeneral, AROONOsc):
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
            
        if (APO[-1] > 0) or (RSIFast[-1] > 50 or RSIFastRising or RSIGeneralRising) or (MACDRising or AROON_direction == 1 or AROON_positive):
            return 1 # Bullish 
            
        elif (APO[-1] < 0) or (RSIFast[-1] < 50 or not RSIFastRising or not RSIGeneralRising) or (not MACDRising or AROON_direction == -1 or not AROON_positive):
            return -1 # Bearish
        else:
            return 0 # Neutral
            
    def rangingFilter(self, ADXR, AROONOsc, MA_same_direction, rsi):
        is_ranging = False
        if ADXR[-1] < 25 or abs(AROONOsc[-1]) < 30 or not MA_same_direction:
            is_ranging = True
        return is_ranging
    
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
        fast, mid, slow = inst['arr_fastMA'], inst['arr_midMA'], inst['arr_slowMA']
        all_MA_up, all_MA_down, MA_same_direction = False, False, False
        if len(fast) > 1 and len(mid) > 1 and len(slow) > 1:
            all_MA_up = fast[-1] > fast[-2] and mid[-1] > mid[-2] and slow[-1] > slow[-2]
            all_MA_down = fast[-1] < fast[-2] and mid[-1] < mid[-2] and slow[-1] < slow[-2]
            MA_same_direction = all_MA_up or all_MA_down
            
        # ranging filter (to confirm moving sideway)
        adxr = talib.ADXR(inst['high_price'], inst['low_price'], inst['arr_close'], 
            timeperiod=self.general_period)
        apo = talib.APO(inst['arr_close'], self.midperiod, self.slowperiod)
        macd, signal, hist = talib.MACD(inst['arr_close'], self.fastperiod, self.slowperiod, self.midperiod)
        rsiFast, rsiGeneral = talib.RSI(inst['arr_close'], self.fastperiod), talib.RSI(inst['arr_close'], self.general_period)       
        # Calculate Aroon values
        aroon_up, aroon_down = talib.AROON(inst['high_price'], inst['low_price'], timeperiod=self.general_period)
        aroonosc = aroon_up - aroon_down
        
        
        # Entry signal 2: stoch RSI crossover
        
        # Long Entry: K crossover D from below
        long_stoch_rsi = inst['K'][-1] > inst['D'][-1] and inst['K'][-2] < inst['D'][-1]
        # Short Entry: K crossover D from above
        short_stoch_rsi = inst['K'][-1] < inst['D'][-1] and inst['K'][-2] > inst['D'][-2]
        
        

        # TODO:  classify the different type of entry signal and set take profit/ stop loss accordingly
        
        ranging = self.rangingFilter(adxr, aroonosc, MA_same_direction, rsiGeneral)
        
        bullish = self.momentumFilter(apo, macd, rsiFast, rsiGeneral, aroonosc)
        
        
        # check for sell signal (price crosses upper bband and rsi > 70)
        if bullish == -1:
            if lastprice >= upper_bband and rsiGeneral[-1] > 70 and ranging:
                return -1
            elif squeeze_breakdown and not ranging:
                return -2
            elif short_stoch_rsi and not ranging:
                return -3 
        # check for buy signal (price crosses lower bband and rsi < 30)
        elif bullish == 1:
            if lastprice <= lower_bband and rsiGeneral[-1] < 30 and ranging:
                return 1
            elif squeeze_breakout and not ranging:
                return 2
            elif long_stoch_rsi and not ranging:
                return 3
        # no signal
        return 0 
      
        
    # execute the trading strat for one instructment given the key and bd       
    def execute_strat(self, bd, key):
        self.evt.consoleLog("---------------------------------")
        self.evt.consoleLog("Executing strat")

        # debug
        self.evt.consoleLog(f"name of instrument: { bd[key]['instrument'] }")
        #self.evt.consoleLog(f"datetime: {bd[self.myinstrument]['timestamp']}")
        #self.evt.consoleLog(f"upper: {upper_bband}")
        #self.evt.consoleLog(f"lower: {lower_bband}")
        #self.evt.consoleLog(f"bbw: {bbw}")
        
        inst =  self.inst_data[key]
        lastprice =  inst['arr_close'][-1]
        # set direction, ie decide if buy or sell, based on entry signal
        direction = 1
        if inst['entry_signal'] > 0:
            direction = 1 #long
        elif inst['entry_singal'] < 0:
            direction = -1 #short
            
        # caclulate the rsi
        
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
            
        self.test_sendOrder(lastprice, direction, 'open', stoploss, takeprofit, self.find_positionSize(lastprice), key, inst['entry_signal'] )
                
        self.evt.consoleLog("Executed strat")
        self.evt.consoleLog("---------------------------------")

    def test_sendOrder(self, lastprice, buysell, openclose, stoploss, takeprofit, volume, instrument, orderRef):
        order = AlgoAPIUtil.OrderObject()
        order.instrument = instrument
        order.orderRef = 1
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
