a model train run similar to those in ../pipelines


but since we don't know which encoder and readout is best, we decide to use all of them


for the encoder: the principle is every signal should be normalized, so we have the following techniques for dealing with int data, whose logarithmic span is about 10
    1. quantize.
        we calculate every 1/128th or 1/256th percentile of the distribution and use a token for each quantile
        this loses adjacency prior and have rounding error

    2. cent quantize.
        using raw cent delta, utilizing the fact that price is inherently quantized to a cent.
        we use this only when value to encode is small (abs <64 cents). we use separate tokens for +inf and -inf otherwise.

    3. draw a graph
        since we are dealing with stock data, a version of CNN over ./draw.py is used by many and worth a try

    4. sin encoding
        like original pos-enc, but for other real numbers


prior to encoding, we need to preprocess for more signals.
originally (~/h/data/a_1min.pq), data is ohlc + volumn for each id and minute.

we want to preserve both original value and relative (e.g. return), so to start with we use:

    1. raw value
    2. per-stock normalized values
        use per-stock average and std of the training set to normalize
    3. return
        over varied periods, 1 min, 30min, since half-day start, or longer.

        over 1min and 30min also use delta in place of ratio, since cent-quantize should work well.

        intra-minute data is highly correlated and should use relative (e.g. close / open, high / open, etc.). cent-quantize should also help here
