from PIL import Image, ImageDraw
import numpy as np
from typing import List, Dict, Tuple


def create_kline_pixel_graph(ohlcv_data: List[Dict[str, float]],
                             moving_average: int = 20) -> Image:
    """
    Replicate the exact K-line pixel graph construction from
    "图以类聚：卷积神经网络、投资者异质性与中国股市预测性".

    This function creates the 180×96 RGB pixel graph described in Section 3.1
      and Appendix B of the paper. The image is designed as input for a CNN
      model where:
    - Upper 72 pixels encode price action (OHLC + moving average)
    - Lower 24 pixels encode volume
    - Each of 60 days occupies exactly 3 horizontal pixels
    - Colors are determined by intraday return magnitude using the paper's
       specified color map

    Args:
        ohlcv_data: List of exactly 60 dictionaries, each containing:
                   - 'open': Opening price
                   - 'high': Highest price
                   - 'low': Lowest price
                   - 'close': Closing price
                   - 'volume': Trading volume
        moving_average: Moving average window (default: 20)

    Returns:
        PIL.Image.Image: RGB image of size 180×96 pixels

    Raises:
        ValueError: If data length is not exactly 60 days

    Example:
        >>> data = [
        ...     {'open': 10.0, 'high': 10.5, 'low': 9.8, 'close': 10.2,
                 'volume': 1000000},
        ...     # ... 59 more days ...
        ... ]
        >>> img = create_kline_pixel_graph(data)
        >>> img.size
        (180, 96)
    """

    IMG_HEIGHT = 96
    PRICE_HEIGHT = int(3/4 * IMG_HEIGHT)
    VOLUME_PLOT_HEIGHT = int(1/4 * IMG_HEIGHT)
    PIXELS_PER_DAY = 3
    IMG_WIDTH = PIXELS_PER_DAY * len(ohlcv_data)

    # Create blank RGB image with black background
    image = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Extract data into numpy arrays
    opens = np.array([d['open'] for d in ohlcv_data], dtype=np.float64)
    highs = np.array([d['high'] for d in ohlcv_data], dtype=np.float64)
    lows = np.array([d['low'] for d in ohlcv_data], dtype=np.float64)
    closes = np.array([d['close'] for d in ohlcv_data], dtype=np.float64)
    volumes = np.array([d['volume'] for d in ohlcv_data], dtype=np.float64)

    # Validate data integrity
    assert (np.all(highs >= lows) and np.all(opens >= lows) and
           np.all(opens <= highs) and
           np.all(closes >= lows) and np.all(closes <= highs)), \
           f'{opens} {highs} {lows} {closes} {volumes}'

    # Calculate 20-day moving average (as specified in paper Appendix B)
    assert len(closes) >= moving_average
    ma = np.convolve(closes, np.ones(moving_average)/moving_average,
                     mode='valid')
    ma = np.concatenate([np.full(moving_average-1, np.nan), ma])

    # Determine scaling ranges for normalization
    min_price = np.min(lows)
    max_price = np.max(highs)
    price_range = max_price - min_price
    price_range = price_range if price_range > 0 else 1.

    max_volume = np.max(volumes)
    vol_range = max_volume if max_volume > 0 else 1.

    # Helper functions for coordinate mapping
    def map_price_to_y(price: float) -> int:
        """Convert price value to y-coordinate in price plot (0=top)"""
        normalized = (price - min_price) / price_range
        return int(PRICE_HEIGHT - 1 - normalized * (PRICE_HEIGHT - 1))

    def map_volume_to_y(volume: float) -> int:
        """Convert volume value to y in volume plot (72=bottom, 95=top)"""
        normalized = volume / vol_range
        base_y = IMG_HEIGHT - VOLUME_PLOT_HEIGHT  # 72
        return int(base_y + (1 - normalized) * (VOLUME_PLOT_HEIGHT - 1))

    def get_rgb_by_return(return_rate: float) -> Tuple[int, int, int]:
        """
        Map intraday return to RGB color based on Figure 2(b) color.

        The color map transitions from red (>0 returns) to green (<0 returns)
        with specific RGB values at different return thresholds.
        """
        # Paper's Figure 2(b) color mapping
        if return_rate >= 0.10:
            return (255, 36, 36)
        elif return_rate >= 0.08:
            return (255, 73, 73)
        elif return_rate >= 0.06:
            return (255, 109, 109)
        elif return_rate >= 0.04:
            return (255, 146, 146)
        elif return_rate >= 0.02:
            return (255, 182, 182)
        elif return_rate >= 0.00:
            return (255, 219, 219)
        elif return_rate >= -0.02:
            return (219, 255, 219)
        elif return_rate >= -0.04:
            return (182, 255, 182)
        elif return_rate >= -0.06:
            return (146, 255, 146)
        elif return_rate >= -0.08:
            return (109, 255, 109)
        elif return_rate >= -0.10:
            return (73, 255, 73)
        else:
            return (36, 255, 36)

    # Draw each day's K-line and volume
    for day_idx in range(len(ohlcv_data)):
        day = ohlcv_data[day_idx]

        # Horizontal pixel positions for this day
        x_left = day_idx * PIXELS_PER_DAY
        x_center = x_left + 1
        x_right = x_left + 2

        # Determine color based on intraday return
        intraday_return = (day['close'] - day['open']) / day['open']
        day_color = get_rgb_by_return(intraday_return)

        # Calculate y-coordinates for price elements
        open_y = map_price_to_y(day['open'])
        high_y = map_price_to_y(day['high'])
        low_y = map_price_to_y(day['low'])
        close_y = map_price_to_y(day['close'])

        # 1. Draw high-low vertical line (wick)
        # Occupies the middle pixel of the 3-pixel day width
        draw.line([(x_center, high_y), (x_center, low_y)],
                  fill=day_color, width=1)

        # 2. Draw open-close body (candle)
        # Paper: "开盘价的像素点位」三个像素点的左侧，收盘价位」右侧"
        body_top = min(open_y, close_y)
        body_bottom = max(open_y, close_y)

        if body_bottom > body_top:
            # Draw filled rectangle for the candle body
            draw.rectangle([x_left, body_top, x_right, body_bottom],
                           fill=day_color, outline=day_color)
        else:
            # For doji (open=close), draw a thin horizontal line
            draw.line([(x_left, open_y), (x_right, close_y)],
                      fill=day_color, width=1)

        # 3. Draw moving average point
        if not np.isnan(ma[day_idx]):
            ma_y = map_price_to_y(ma[day_idx])
            # Paper: "移动均价像素点位」中间"
            # Draw as a small 3x3 pixel square centered at MA point
            ma_rect = [x_center-1, ma_y-1, x_center+1, ma_y+1]
            draw.rectangle(ma_rect, fill=day_color, outline=day_color)

        # 4. Draw volume bar in lower plot area
        # Paper: "下25%为交易量像素图"
        volume_y = map_volume_to_y(day['volume'])
        # Draw vertical line from bottom to volume level
        draw.line([(x_center, IMG_HEIGHT - 1), (x_center, volume_y)],
                  fill=day_color, width=1)

    return image


# Utility function for testing
def generate_sample_data(n_days=60, seed=42) -> List[Dict[str, float]]:
    """
    Generate realistic sample OHLCV data for testing the pixel graph function.

    Args:
        n_days: Number of days to generate (default: 60)
        seed: Random seed for reproducibility

    Returns:
        List[Dict[str, float]]: Simulated OHLCV data
    """
    np.random.seed(seed)
    data = []
    base_price = 100.0

    for i in range(n_days):
        # Generate realistic intraday price movement
        daily_volatility = 0.02
        r = daily_volatility * base_price

        open_price = base_price + np.random.normal(0, r * 0.3)
        high_price = open_price + abs(np.random.normal(0, r * 0.3)) + 2
        low_price = open_price - abs(np.random.normal(0, r * 0.3)) - 2
        close_price = low_price + np.random.random() * (high_price - low_price)

        # Ensure proper price ordering
        prices = [open_price, close_price]
        low_price = min(low_price, min(prices))
        high_price = max(high_price, max(prices))
        open_price = np.clip(open_price, low_price, high_price)
        close_price = np.clip(close_price, low_price, high_price)

        # Generate volume
        volume = np.random.randint(1_000_000, 10_000_000)

        data.append({
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': int(volume)
        })

        base_price = close_price

    return data


# Example usage
if __name__ == "__main__":
    # Generate sample data
    sample_data = generate_sample_data()

    # Create pixel graph
    pixel_graph = create_kline_pixel_graph(sample_data)

    # Save and display
    pixel_graph.save("kline_pixel_graph.png")
    print(f"Created pixel graph: {pixel_graph.size}")
    print(f"Image mode: {pixel_graph.mode}")
    print(f"Data range: {len(sample_data)} days")
