# ET ArthaAI — Autonomous Money Mentor

ET ArthaAI is a premium, AI-powered financial intelligence platform built for Indian investors. It features a sleek, dark-luxury Streamlit interface combined with a CrewAI multi-agent system powered by Google's Gemini 1.5 Flash.

The system acts as your personal wealth mentor, dynamically analyzing your financial profile to calculate early retirement (FIRE) goals, compare tax regimes (FY 2024-25), and identify hidden risks in your mutual fund portfolio.

## 🌟 Key Features

* **🔥 FIRE Blueprint:** Calculates your inflation-adjusted Financial Independence, Retire Early (FIRE) corpus, projects accumulation timelines, and suggests the required monthly SIP.
* **🧾 Tax Auditor:** Compares the Old and New Tax Regimes step-by-step for FY 2024-25, highlighting missed deductions (80C, HRA, NPS) and recommending the optimal path.
* **🔍 Portfolio X-Ray:** Analyzes stock overlaps across your selected mutual funds to flag high concentration risks (e.g., heavy reliance on specific large-cap stocks) to improve diversification.
* **🤖 Autonomous AI Agents:** Utilizes CrewAI to orchestrate three specialized agents:
  1. *Retirement Strategist*
  2. *Tax Auditor*
  3. *Portfolio X-Ray Analyst*
* **📊 Interactive Visualizations:** Beautiful Plotly-powered charts tracking wealth journeys, tax distributions, and stock overlaps.

## 🛠️ Technology Stack

* **Frontend:** Streamlit
* **AI Orchestration:** CrewAI
* **LLM:** Google Gemini 1.5 Flash (`langchain-google-genai`)
* **Data & Vis:** Pandas, Plotly Express, Plotly Graph Objects

## 🚀 How to Run Locally

### Prerequisites

Ensure you have Python 3.9+ installed.

### 1. Install Dependencies

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

### 3. Start the Application

You can launch the main Streamlit application by running:

```bash
streamlit run app.py
```
*(Alternatively, you can run `streamlit run frondend.py` for the tier-1 fintech aesthetic variant).*

## 📂 Project Structure

* `app.py`: Main single-file Streamlit application with the luxury dark theme and CrewAI logic.
* `frondend.py`: Another Streamlit variant with an enterprise-prestige aesthetic.
* `main.py`: Core backend logic and LLM architecture.
* `requirements.txt`: Python dependencies required to run the platform.

Drive link:https://drive.google.com/drive/folders/1JpLgpxV1tdPPfnRN6CJy3yp5vgXmXVMg?usp=sharing
 
DRIVE LINK HAS(Pitch Video, Architecture, Business Impact Model, Presentation)
