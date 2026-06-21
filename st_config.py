# st_config.py
import os

# -------------------------------
# ST GLOBAL CONFIG / SINGLE SOURCE
# -------------------------------

DEBUG_ST = False

COMMON_DIR = r"C:\Users\Uzair Khan\AppData\Roaming\MetaQuotes\Terminal\Common\Files"

ST_REQUEST_FILE = "st_request.txt"
ST_DONE_FILE = "st_done.txt"
ST_ERROR_FILE = "st_error.txt"
ST_SUBFOLDER = os.path.join("STBridge")

# ---- SuperTrend_TV_Same.mq5 aligned defaults ----
ST_PERIODS = 5
ST_MULTIPLIER = 1.25
ST_SOURCE = "HL2"          # matches: src = SRC_HL2
ST_CHANGE_ATR = True       # matches: changeATR = true
ST_SHOW_SIGNALS = True     # matches: showsignals = true
ST_ENABLE_ALERTS = True    # matches: enableAlerts = true

# bridge / export settings
ST_BARS_TO_EXPORT = 300
ST_TIMEOUT_SEC = 30

# default timeframe settings
ST_DEFAULT_TIMEFRAME = "M15"
ST_DEFAULT_TIMEFRAMES = ["M15"]
