import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

# --- 核心逻辑 (保持不变) ---
# --- 修复乱码代码开始 ---
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题
# --- 修复乱码代码结束 ---

class Aggressor(Agent):
    def __init__(self, unique_id, model, strength, decay_rate):
        super().__init__(unique_id, model)
        self.strength = strength
        self.morale = 100.0
        self.decay_rate = decay_rate

    def step(self):
        target = self.model.get_nearest_fabian(self.pos)
        if target:
            self.model.move_towards(self, target.pos, speed=1.2)
        self.morale -= self.decay_rate
        if self.morale < 0: self.morale = 0

class FabianAgent(Agent):
    def __init__(self, unique_id, model, safe_dist, harass_power):
        super().__init__(unique_id, model)
        self.safe_dist = safe_dist
        self.harass_power = harass_power

    def step(self):
        enemy = self.model.get_nearest_aggressor(self.pos)
        if not enemy: return
        dist = self.model.space.get_distance(self.pos, enemy.pos)

        if enemy.morale < 30: 
            self.model.move_towards(self, enemy.pos, speed=1.5)
            enemy.morale -= self.harass_power * 2
        elif dist < self.safe_dist: 
            self.model.move_away(self, enemy.pos, speed=1.5)
        elif dist > self.safe_dist + 5: 
            self.model.move_towards(self, enemy.pos, speed=1.0)
        
        if dist < self.safe_dist * 1.5:
            enemy.morale -= self.harass_power

class StrategyModel(Model):
    def __init__(self, n_aggressors, n_fabians, safe_dist, decay_rate, harass_power):
        super().__init__()
        self.space = ContinuousSpace(100, 100, False)
        self.schedule = RandomActivation(self)
        for i in range(n_aggressors):
            a = Aggressor(i, self, strength=100, decay_rate=decay_rate)
            self.schedule.add(a)
            self.space.place_agent(a, (np.random.uniform(0, 20), np.random.uniform(0, 20)))
        for i in range(n_fabians):
            f = FabianAgent(i + 100, self, safe_dist=safe_dist, harass_power=harass_power)
            self.schedule.add(f)
            self.space.place_agent(f, (np.random.uniform(40, 60), np.random.uniform(40, 60)))
        self.datacollector = DataCollector({
            "Morale/Capital": lambda m: np.mean([a.morale for a in m.schedule.agents if isinstance(a, Aggressor)])
        })

    def move_towards(self, agent, target_pos, speed):
        curr_x, curr_y = agent.pos
        tx, ty = target_pos
        dx, dy = tx - curr_x, ty - curr_y
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
        # 计算原始新坐标
            new_x = curr_x + dx/dist * speed
            new_y = curr_y + dy/dist * speed
        
        # --- 核心修复：添加边界裁剪 (Clipping) ---
            new_x = max(0, min(99.9, new_x))
            new_y = max(0, min(99.9, new_y))
        
            new_pos = (new_x, new_y)
            self.space.move_agent(agent, new_pos)

    def move_away(self, agent, target_pos, speed):
        curr_x, curr_y = agent.pos
        tx, ty = target_pos
        dx, dy = tx - curr_x, ty - curr_y
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            new_pos = (curr_x - dx/dist * speed, curr_y - dy/dist * speed)
            new_pos = (max(0, min(99.9, new_pos[0])), max(0, min(99.9, new_pos[1])))
            self.space.move_agent(agent, new_pos)

    def get_nearest_fabian(self, pos):
        agents = [a for a in self.schedule.agents if isinstance(a, FabianAgent)]
        return self._get_min_dist(pos, agents)

    def get_nearest_aggressor(self, pos):
        agents = [a for a in self.schedule.agents if isinstance(a, Aggressor)]
        return self._get_min_dist(pos, agents)

    def _get_min_dist(self, pos, agents):
        if not agents: return None
        distances = [self.space.get_distance(pos, a.pos) for a in agents]
        return agents[np.argmin(distances)]

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# --- Streamlit UI 与 多视角映射 ---
st.sidebar.image(
    "https://assets.zyrosite.com/cdn-cgi/image/format=auto,w=768,fit=crop,q=95/1evUiS818YahKfZE/pythonlogo2-AfiMET3ydIQjjfId.png", 
    use_container_width=True
)

# 2. 加上实验室名称与标语
st.sidebar.title("Python历史战略实验室")
st.set_page_config(page_title="Python历史战略实验室", layout="wide")


# 视角切换
view_mode = st.sidebar.radio("选择推演视角", ["历史战略视角", "MBA 商业视角"])

# 根据视角定义字典
if view_mode == "历史战略视角":
    mapping = {
        "title": "⚔️ 费边策略：第二次布匿战争模拟",
        "aggressor_name": "汉尼拔远征军",
        "fabian_name": "费边防御部队",
        "resource_name": "军队士气/粮草",
        "decay_label": "远征后勤压力 (Attrition)",
        "safe_label": "地理缓冲区深度",
        "harass_label": "游击骚扰强度",
        "success_msg": "观察士气曲线：当士气归零，意味着费边通过空间换取了时间，罗马赢得了最终胜利。"
    }
else:
    mapping = {
        "title": "📊 颠覆性创新：巨头 vs. 挑战者",
        "aggressor_name": "行业现任巨头 (Incumbent)",
        "fabian_name": "敏捷颠覆者 (Disruptor)",
        "resource_name": "可用资本/市场份额",
        "decay_label": "组织运营烧钱率 (Burn Rate)",
        "safe_label": "蓝海策略隔离带",
        "harass_label": "利润蚕食能力",
        "success_msg": "观察资本曲线：当资本归零，意味着巨头因无法应对非对称竞争而退出市场。"
    }

st.title(mapping["title"])
st.sidebar.markdown(f"### 当前模式：{view_mode}")

with st.sidebar:
    st.header("战场/市场参数")
    s_dist = st.slider(mapping["safe_label"], 5, 30, 15)
    d_rate = st.slider(mapping["decay_label"], 0.1, 1.0, 0.3)
    h_pow = st.slider(mapping["harass_label"], 0.05, 0.5, 0.2)
    steps = st.slider("模拟周期", 50, 300, 150)

if st.button("开始战略推演"):
    model = StrategyModel(1, 3, s_dist, d_rate, h_pow)
    col1, col2 = st.columns([2, 1])
    plot_spot = col1.empty()
    chart_spot = col2.empty()

    for i in range(steps):
        model.step()
        fig, ax = plt.subplots(figsize=(5, 5))
        for agent in model.schedule.agents:
            x, y = agent.pos
            color = 'red' if isinstance(agent, Aggressor) else 'blue'
            marker = 'X' if isinstance(agent, Aggressor) else 'o'
            ax.scatter(x, y, c=color, marker=marker, s=100)
            if isinstance(agent, Aggressor):
                ax.text(x, y+3, f"{mapping['resource_name']}: {agent.morale:.1f}", ha='center', fontsize=8)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_title(f"回合/季度: {i}")
        plot_spot.pyplot(fig)
        plt.close(fig)

        df = model.datacollector.get_model_vars_dataframe()
        chart_spot.line_chart(df)

    st.success(mapping["success_msg"])

    # --- 深度分析报告 ---
    st.markdown("---")
    st.subheader(f"📑 {view_mode} 战略复盘")
    final_val = df.iloc[-1]["Morale/Capital"]
    
    if view_mode == "历史战略视角":
        if final_val > 50:
            st.write("**战局评论**：汉尼拔维持了强大的存在。费边的拖延未能动摇远征军根基，罗马面临决战压力。")
        else:
            st.write("**战局评论**：费边策略大获全胜。汉尼拔在漫长的周旋中耗尽了最后一斗米，罗马不战而胜。")
        st.info("💡 **历史映射**：此模型体现了‘空间换时间’的核心逻辑。")
    else:
        if final_val > 50:
            st.write("**商战评论**：巨头凭借深厚的护城河守住了阵地。初创企业的‘骚扰’未能触及核心盈利业务。")
        else:
            st.write("**商战评论**：典型的‘创新者窘境’。巨头被高昂的运营成本和初创企业的侧翼蚕食拖垮。")

        st.info("💡 **MBA 映射**：此模型体现了‘破坏性创新’如何利用大企业的固定成本优势进行反向打击。")
