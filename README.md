# 🏛️ Python Historical Strategy Laboratory

**Fabian Strategy Simulator** — Second Punic War × MBA Disruptive Innovation Dual-Perspective War Room

An interactive strategy sandbox built with **Mesa Agent-Based Modeling** + **Streamlit**.  
Adjust parameters in real time and watch “Hannibal’s Expedition” get worn down by Fabius… or see an industry incumbent collapse under a nimble disruptor.

---

## ✨ Key Features

- **Seamless Dual-Perspective Switching**  
  - **Historical Strategy View**: Hannibal (Aggressor) vs Fabian Defense Force  
  - **MBA Business View**: Incumbent Giant vs Agile Disruptor

- **Live 2D Battlefield Visualization**  
  Red X = Hannibal/Incumbent (with real-time morale/capital label)  
  Blue O = Fabian/Disruptor

- **Dynamic Morale / Capital Line Chart** (Streamlit native)

- **Real-time Parameter Controls** (sidebar sliders):
  - Geographic Buffer Depth / Blue-Ocean Isolation Zone
  - Expedition Logistics Pressure / Organizational Burn Rate
  - Guerrilla Harassment Intensity / Profit Erosion Power
  - Simulation Cycles (50–300 steps)

- **Smart After-Action Report** with historical/MBA mapping

- **Boundary Clipping Fix** — agents can no longer fly off the map

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install streamlit pandas numpy matplotlib mesa
