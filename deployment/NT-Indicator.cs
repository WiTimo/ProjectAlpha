#region Using declarations
using System;
using System.Globalization;
using System.IO;
using NinjaTrader.Data;
using NinjaTrader.Gui.NinjaScript;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class L2FileLogger : Indicator
    {
        private StreamWriter writer;
        private readonly object fileLock = new object();

        [NinjaScriptProperty]
        public string FilePath { get; set; }

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "_L2FileLogger";
                Description = "Logs Level II updates to a text file.";
                IsOverlay = false;
                Calculate = Calculate.OnEachTick;
                IsSuspendedWhileInactive = true;

                FilePath = "C:\\Ninjatrader\\L2Log.txt";
            }
            else if (State == State.DataLoaded)
            {
                OpenWriter();
            }
            else if (State == State.Terminated)
            {
                CloseWriter();
            }
        }

        private void OpenWriter()
        {
            try
            {
                if (string.IsNullOrWhiteSpace(FilePath))
                    return;

                string dir = Path.GetDirectoryName(FilePath);
                if (!Directory.Exists(dir))
                    Directory.CreateDirectory(dir);

                writer = new StreamWriter(FilePath, true)
                {
                    AutoFlush = true
                };
            }
            catch (Exception ex)
            {
                Print("ERROR opening L2 log file: " + ex.Message);
            }
        }

        private void CloseWriter()
        {
            try
            {
                if (writer != null)
                {
                    writer.Flush();
                    writer.Close();
                    writer.Dispose();
                }
            }
            catch { }
        }

        protected override void OnMarketDepth(MarketDepthEventArgs e)
        {
            if (State != State.Realtime || writer == null)
                return;

            try
            {
                DateTime now = DateTime.Now;

                string timestamp = now.ToString("yyyyMMddHHmmss", CultureInfo.InvariantCulture);

                // correct long → int cast
                int offset100ns = (int)(now.Ticks % TimeSpan.TicksPerSecond);

                int type = (int)e.MarketDataType;
                int operation = (int)e.Operation;
                int position = e.Position;

                string mm = e.MarketMaker ?? "";
                string price = e.Price.ToString(CultureInfo.CurrentCulture);
                int volume = (int) e.Volume;

                string line = string.Format(
                    "L2;{0};{1};{2};{3};{4};{5};{6};{7}",
                    type, timestamp, offset100ns, operation, position, mm, price, volume
                );

                lock (fileLock)
                    writer.WriteLine(line);
            }
            catch (Exception ex)
            {
                Print("L2FileLogger ERROR: " + ex.Message);
            }
        }
    }
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private L2FileLogger[] cacheL2FileLogger;
		public L2FileLogger L2FileLogger(string filePath)
		{
			return L2FileLogger(Input, filePath);
		}

		public L2FileLogger L2FileLogger(ISeries<double> input, string filePath)
		{
			if (cacheL2FileLogger != null)
				for (int idx = 0; idx < cacheL2FileLogger.Length; idx++)
					if (cacheL2FileLogger[idx] != null && cacheL2FileLogger[idx].FilePath == filePath && cacheL2FileLogger[idx].EqualsInput(input))
						return cacheL2FileLogger[idx];
			return CacheIndicator<L2FileLogger>(new L2FileLogger(){ FilePath = filePath }, input, ref cacheL2FileLogger);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.L2FileLogger L2FileLogger(string filePath)
		{
			return indicator.L2FileLogger(Input, filePath);
		}

		public Indicators.L2FileLogger L2FileLogger(ISeries<double> input , string filePath)
		{
			return indicator.L2FileLogger(input, filePath);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.L2FileLogger L2FileLogger(string filePath)
		{
			return indicator.L2FileLogger(Input, filePath);
		}

		public Indicators.L2FileLogger L2FileLogger(ISeries<double> input , string filePath)
		{
			return indicator.L2FileLogger(input, filePath);
		}
	}
}

#endregion
