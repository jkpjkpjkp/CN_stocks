
import polars as pl
from datetime import date, timedelta
def proc(df):
    start_price = df['close'][0]
    start_datetime = df['datetime'][0]
    assert df['id'][0] == df['id'][-1]
    
    base_freq = df.select((pl.col.datetime - pl.col.datetime.shift(1)).mode())
    
    out = [{'timestamp': start_datetime.timestamp(), 'start_price': start_price, 'id': df['id'][0]}]
    
    ret = []
    
    last_close = start_price
    last_datetime = start_datetime
    for x in df.iter_rows(named=True)[1:]:
        delta = int((x['close'] - last_close) * 100 + 0.1)
        duration = (x['datetime'] - last_datetime).total_seconds()
        last_close = x['close']
        last_datetime = x['datetime']
        
        if duration < base_freq:
            out[-1] = delta
            continue
        elif duration == base_freq:
            pass
        elif duration < timedelta(hours=1):
            out.extend([0] * (duration // base_freq - 1))
        elif duration < timedelta(hours=4):
            out.append("LUNCH")
        elif duration < timedelta(days=1):
            out.append("DINNER")
        elif duration < timedelta(days=3):
            out.append("WEEKEND")
        elif duration < timedelta(days=8):
            out.append("SHORT HOLIDAY")
        elif duration < timedelta(days=31):
            out.append("HOLIDAY")
        else:
            ret.append(out)
            out = [{'timestamp': x['datetime'].timestamp(), 'start_price': x['close'], 'id': x['id']}]

        out.append(delta)
    
    ret.append(out)
    return ret

# this is for a single df. i have large, chunked df_s, should group by id and apply this to all id in df_s, and concat the `ret`


def large_number_tokenizer(arr: list):
    ret = [arr[0]]
    for x in arr[1:]:
        if isinstance(x, str):
            ret.append(x)
        elif abs(x)< 64:
            ret.append(x)
        else:
            # get the bit-encoding of x+y, where y is the smallest pow-of-2 to make x+y nonnegative
            y = 2**(x.bit_length() - 1)
            
            # first, append (x + y) mod 128
            ret.append((x + y) % 128)
            
            # then, bit by bit append higher bits of x+y
            for i in range(7, x.bit_length()):
                ret.append(str((x+y & 1 << i) - (1 << i+1)))