import numpy as np
from .agents.adaptive_agent import AdaptiveAgent
from .environment.environment import Environment
from .interface.visualization import visualize_results
from .config.parameters import *


def simulate_advanced(episodes=800, initial_num_agents=20):
    env = Environment()
    agents = [AdaptiveAgent(agent_id=i) for i in range(initial_num_agents)]
    all_critical_episodes = {}
    social_interactions = []
    level_up_episodes = {}

    # Инициализируем статистику для всех агентов
    for agent in agents:
        all_critical_episodes[agent.id] = []
        level_up_episodes[agent.id] = []

    activation_stats = {}
    reproduction_events = []
    death_events = []
    cooperation_events = []
    next_agent_id = initial_num_agents

    for episode in range(episodes):
        # Удаляем мертвых агентов
        agents_to_remove = []
        for agent in agents:
            if agent.check_death_conditions():
                agent.alive = False
                death_events.append((episode, agent.id, 'natural'))
                agents_to_remove.append(agent)
                print(f"Агент {agent.id} умер на эпизоде {episode}")

        agents = [agent for agent in agents if agent not in agents_to_remove]

        social_interaction_strength = 0
        new_agents = []

        for agent in agents:
            if not agent.alive:
                continue

            state = np.random.randint(env.num_states)
            other_agents = [a for a in agents if a.id != agent.id and a.alive]
            action = agent.choose_action(state, other_agents)
            reward = env.get_reward(state, action, episode, agent.id)
            next_state = (state + np.random.randint(1, 3)) % env.num_states

            # Кооперативные взаимодействия
            if other_agents and np.random.random() < 0.3:
                for other in np.random.choice(other_agents, size=min(2, len(other_agents)), replace=False):
                    cooperation_result = agent.cooperate(other)
                    if cooperation_result > 0:
                        cooperation_events.append((episode, agent.id, other.id, cooperation_result))

            agent.update(state, action, reward, next_state, other_agents)

            # Проверяем метацели
            new_activations = agent.check_meta_goals(episode)
            if new_activations:
                all_critical_episodes[agent.id].extend([(episode, meta) for meta in new_activations])
                for meta in new_activations:
                    if agent.id not in activation_stats:
                        activation_stats[agent.id] = {'self': [], 'pred': [], 'exp': []}
                    activation_stats[agent.id][meta].append(episode)
                print(f"Агент {agent.id}: метацели активированы на эпизоде {episode}: {new_activations}")

            if agent.level_up_episode == episode:
                level_up_episodes[agent.id].append(episode)

            # Проверяем размножение
            if agent.check_reproduction_conditions():
                new_agent = agent.reproduce(next_agent_id)
                new_agents.append(new_agent)
                reproduction_events.append((episode, agent.id, next_agent_id))
                print(f"Агент {agent.id} размножился на эпизоде {episode}, родился агент {next_agent_id}")
                next_agent_id += 1

                social_interaction_strength += LIFE_PARAMS['reproduction_social_influence_bonus']

            # Социальное влияние
            if other_agents:
                social_influence = 0
                for other in other_agents:
                    if other.id != agent.id:
                        social_influence += other.knowledge * SOCIAL_PARAMS['social_influence_factor']
                social_interaction_strength += social_influence

        # Добавляем новорожденных агентов
        agents.extend(new_agents)
        for new_agent in new_agents:
            all_critical_episodes[new_agent.id] = []
            level_up_episodes[new_agent.id] = []
            activation_stats[new_agent.id] = {'self': [], 'pred': [], 'exp': []}

        social_interactions.append(social_interaction_strength)

        if episode % 100 == 0:
            alive_count = len([a for a in agents if a.alive])
            avg_age = np.mean([a.age for a in agents if a.alive]) if alive_count > 0 else 0
            print(
                f"Эпизод {episode}, Агентов: {alive_count}, Средний возраст: {avg_age:.1f}, Социальная активность: {social_interaction_strength:.2f}")

            for agent in agents:
                if not agent.alive:
                    continue
                active_goals = [goal for goal, active in agent.meta_active.items() if active]
                if active_goals:
                    print(f"  Агент {agent.id}: активные цели: {active_goals}, S: {agent.S:.2f}, Возраст: {agent.age}")

        # ДОБАВЛЯЕМ экстренную помощь если популяция вымирает
        if episode > 50 and len([a for a in agents if a.alive]) < 3:
            print(f"ЭКСТРЕННАЯ ПОМОЩЬ: популяция на грани вымирания!")
            for agent in agents:
                if agent.alive:
                    # Даем ресурсы выжившим агентам
                    agent.energy += 10.0
                    agent.territory += 2.0
                    print(f"  Агент {agent.id} получил экстренную помощь")

    # Визуализация и анализ
    visualize_results(agents, all_critical_episodes, social_interactions, episodes,
                      level_up_episodes, reproduction_events, death_events, cooperation_events)

    return {
        'agents': agents,
        'critical_episodes': all_critical_episodes,
        'activation_stats': activation_stats,
        'reproduction_events': reproduction_events,
        'death_events': death_events,
        'cooperation_events': cooperation_events
    }