import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy


def visualize_results(agents, critical_episodes, social_interactions, episodes, level_up_episodes,
                      reproduction_events, death_events, cooperation_events):
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))

    colors = plt.cm.tab20(np.linspace(0, 1, len(agents)))

    # Фильтруем только живых агентов
    alive_agents = [agent for agent in agents if agent.alive]
    if not alive_agents:
        alive_agents = agents

    # 1. Графики мета-ресурсов
    for i, agent in enumerate(alive_agents):
        color = colors[i % len(colors)]

        # Убедимся, что история имеет правильную длину
        energy_hist = agent.history['energy'][:episodes] if agent.history.get('energy') else [0] * episodes
        knowledge_hist = agent.history['knowledge'][:episodes] if agent.history.get('knowledge') else [0] * episodes
        territory_hist = agent.history['territory'][:episodes] if agent.history.get('territory') else [0] * episodes

        # Заполним недостающие значения последним известным значением или 0
        if len(energy_hist) < episodes:
            last_value = energy_hist[-1] if energy_hist else 0
            energy_hist.extend([last_value] * (episodes - len(energy_hist)))
        if len(knowledge_hist) < episodes:
            last_value = knowledge_hist[-1] if knowledge_hist else 0
            knowledge_hist.extend([last_value] * (episodes - len(knowledge_hist)))
        if len(territory_hist) < episodes:
            last_value = territory_hist[-1] if territory_hist else 0
            territory_hist.extend([last_value] * (episodes - len(territory_hist)))

        axes[0, 0].plot(energy_hist, label=f'Агент {agent.id}', alpha=0.7, color=color)
        axes[0, 1].plot(knowledge_hist, label=f'Агент {agent.id}', alpha=0.7, color=color)
        axes[0, 2].plot(territory_hist, label=f'Агент {agent.id}', alpha=0.7, color=color)

        # Отмечаем критические эпизоды
        for ep, meta_type in critical_episodes.get(agent.id, []):
            if ep < episodes:  # Проверяем, что эпизод в пределах диапазона
                if meta_type == 'self':
                    axes[0, 0].axvline(x=ep, color=color, linestyle='--', alpha=0.5)
                elif meta_type == 'pred':
                    axes[0, 1].axvline(x=ep, color=color, linestyle='--', alpha=0.5)
                elif meta_type == 'exp':
                    axes[0, 2].axvline(x=ep, color=color, linestyle='--', alpha=0.5)

    # Отмечаем события
    for ep, parent_id, child_id in reproduction_events:
        if ep < episodes:
            axes[0, 0].axvline(x=ep, color='green', linestyle=':', alpha=0.3,
                               label='Размножение' if ep == reproduction_events[0][0] else "")
    for ep, agent_id, cause in death_events:
        if ep < episodes:
            axes[0, 0].axvline(x=ep, color='red', linestyle=':', alpha=0.3,
                               label='Смерть' if ep == death_events[0][0] else "")

    axes[0, 0].set_title('Самосохранение (S_self)')
    axes[0, 1].set_title('Предсказуемость (S_pred)')
    axes[0, 2].set_title('Экспансия (S_exp)')

    # 2. Графики сложности и эффективности
    for i, agent in enumerate(alive_agents):
        color = colors[i % len(colors)]

        # Обеспечиваем правильную длину истории
        complexity_hist = agent.history['complexity'][:episodes] if agent.history.get('complexity') else [0] * episodes
        phi_hist = agent.history['phi'][:episodes] if agent.history.get('phi') else [0] * episodes
        e_eff_hist = agent.history['e_eff'][:episodes] if agent.history.get('e_eff') else [0] * episodes

        if len(complexity_hist) < episodes:
            last_value = complexity_hist[-1] if complexity_hist else 0
            complexity_hist.extend([last_value] * (episodes - len(complexity_hist)))
        if len(phi_hist) < episodes:
            last_value = phi_hist[-1] if phi_hist else 0
            phi_hist.extend([last_value] * (episodes - len(phi_hist)))
        if len(e_eff_hist) < episodes:
            last_value = e_eff_hist[-1] if e_eff_hist else 0
            e_eff_hist.extend([last_value] * (episodes - len(e_eff_hist)))

        axes[1, 0].plot(complexity_hist, label=f'Агент {agent.id}', alpha=0.7, color=color)
        axes[1, 1].plot(phi_hist, label=f'Агент {agent.id}', alpha=0.7, color=color)
        axes[1, 2].plot(e_eff_hist, label=f'Агент {agent.id}', alpha=0.7, color=color)

    axes[1, 0].set_title('Сложность (C)')
    axes[1, 1].set_title('Интегрированная информация (Φ)')
    axes[1, 2].set_title('Эволюционная эффективность (E_eff)')

    # 3. Социальная активность и исследование
    social_hist = social_interactions[:episodes] if social_interactions else [0] * episodes
    if len(social_hist) < episodes:
        last_value = social_hist[-1] if social_hist else 0
        social_hist.extend([last_value] * (episodes - len(social_hist)))

    axes[2, 0].plot(social_hist, label='Социальная активность', color='purple')
    axes[2, 0].set_title('Социальные взаимодействия')
    axes[2, 0].set_xlabel('Эпизод')

    # Динамика исследования
    exploration_rates = []
    for agent in alive_agents:
        exp_hist = agent.history['exploration_rate'][:episodes] if agent.history.get('exploration_rate') else [
                                                                                                                  0] * episodes
        if len(exp_hist) < episodes:
            last_value = exp_hist[-1] if exp_hist else 0
            exp_hist.extend([last_value] * (episodes - len(exp_hist)))
        exploration_rates.append(exp_hist)

    for i, rates in enumerate(exploration_rates):
        axes[2, 1].plot(rates, label=f'Агент {alive_agents[i].id}', alpha=0.7, color=colors[i % len(colors)])
    axes[2, 1].set_title('Динамика исследования')
    axes[2, 1].set_xlabel('Эпизод')
    axes[2, 1].legend(fontsize=8)

    # 4. Динамика порядка, энтропии и уровня
    for i, agent in enumerate(alive_agents):
        color = colors[i % len(colors)]

        # Обеспечиваем правильную длину истории
        P_hist = agent.history['P'][:episodes] if agent.history.get('P') else [0] * episodes
        S_hist = agent.history['S'][:episodes] if agent.history.get('S') else [0] * episodes
        level_hist = agent.history['level'][:episodes] if agent.history.get('level') else [1] * episodes

        if len(P_hist) < episodes:
            last_value = P_hist[-1] if P_hist else 0
            P_hist.extend([last_value] * (episodes - len(P_hist)))
        if len(S_hist) < episodes:
            last_value = S_hist[-1] if S_hist else 0
            S_hist.extend([last_value] * (episodes - len(S_hist)))
        if len(level_hist) < episodes:
            last_value = level_hist[-1] if level_hist else 1
            level_hist.extend([last_value] * (episodes - len(level_hist)))

        axes[2, 2].plot(P_hist, label=f'Агент {agent.id}', alpha=0.7, color=color)
        axes[3, 0].plot(S_hist, label=f'Агент {agent.id}', alpha=0.7, color=color)
        axes[3, 1].plot(level_hist, label=f'Агент {agent.id}', alpha=0.7, color=color)

        if hasattr(agent,
                   'level_up_episode') and agent.level_up_episode is not None and agent.level_up_episode < episodes:
            axes[3, 1].axvline(x=agent.level_up_episode, color=color, linestyle='--', alpha=0.5)

    axes[2, 2].set_title('Динамика порядка (P)')
    axes[3, 0].set_title('Динамика энтропии (S)')
    axes[3, 1].set_title('Уровень эмерджентности')
    axes[3, 1].set_xlabel('Эпизод')

    # 5. Динамика популяции - ПЕРЕПИСАННАЯ ЛОГИКА
    population_history = []

    # Создаем список всех агентов с их временем жизни
    agent_lifetimes = []

    for agent in agents:
        birth_episode = getattr(agent, 'birth_episode', 0)  # Начальные агенты рождены в эпизоде 0
        death_episode = getattr(agent, 'death_episode', episodes)  # Если не умер, то живет до конца

        agent_lifetimes.append((birth_episode, death_episode))

    # Считаем популяцию для каждого эпизода
    for episode in range(episodes):
        count = 0
        for birth, death in agent_lifetimes:
            # Агент жив в этом эпизоде, если родился до или в этом эпизоде и умер после этого эпизода
            if birth <= episode and death > episode:
                count += 1
        population_history.append(count)

    axes[3, 2].plot(population_history, color='blue', linewidth=2)
    axes[3, 2].set_title('Динамика популяции')
    axes[3, 2].set_xlabel('Эпизод')
    axes[3, 2].set_ylabel('Количество агентов')
    axes[3, 2].grid(True, alpha=0.3)

    # Устанавливаем правильные пределы для оси Y
    if population_history:
        max_population = max(population_history)
        min_population = min(population_history)
        y_max = max(10, max_population * 1.1)  # Минимум 10, чтобы график не был плоским
        y_min = max(0, min_population * 0.9)
        axes[3, 2].set_ylim(y_min, y_max)
    else:
        axes[3, 2].set_ylim(0, 10)

    plt.tight_layout()
    plt.show()

    # Анализ результатов
    print("\n" + "=" * 60)
    print("ПОЛНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ЭВОЛЮЦИОННОЙ СИМУЛЯЦИИ:")
    print("=" * 60)

    # Статистика популяции
    initial_count = len([a for a in agents if not hasattr(a, 'parent_id') or a.parent_id is None])
    final_count = len([a for a in agents if a.alive])
    avg_age = np.mean([a.age for a in agents if a.alive]) if final_count > 0 else 0

    print(f"\nПОПУЛЯЦИЯ:")
    print(f"  Началось с: {initial_count} агентов")
    print(f"  Закончилось с: {final_count} агентов")
    print(f"  Всего рождений: {len(reproduction_events)}")
    print(f"  Всего смертей: {len(death_events)}")
    print(f"  Средний возраст: {avg_age:.1f} эпизодов")

    # Добавляем статистику по динамике популяции
    if population_history:
        print(f"  Максимальная популяция: {max(population_history)}")
        print(f"  Минимальная популяция: {min(population_history)}")
        print(f"  Средняя популяция: {np.mean(population_history):.1f}")

    # Анализ размножения
    reproductive_success = {}
    for _, parent_id, _ in reproduction_events:
        reproductive_success[parent_id] = reproductive_success.get(parent_id, 0) + 1

    if reproductive_success:
        print(f"\nРЕПРОДУКТИВНЫЙ УСПЕХ:")
        for agent_id, count in sorted(reproductive_success.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Агент {agent_id}: {count} потомков")

    # Анализ кооперации
    total_cooperation = sum(
        sum(getattr(agent, 'cooperation_history', {}).values()) for agent in agents if
        hasattr(agent, 'cooperation_history'))
    avg_cooperation = total_cooperation / final_count if final_count > 0 else 0
    print(f"\nКООПЕРАЦИЯ:")
    print(f"  Всего взаимодействий: {total_cooperation}")
    print(f"  В среднем на агента: {avg_cooperation:.1f}")

    # Анализ метацелей
    meta_goal_stats = {'self': 0, 'pred': 0, 'exp': 0}
    for agent in agents:
        if hasattr(agent, 'activation_episodes'):
            for goal in ['self', 'pred', 'exp']:
                if agent.activation_episodes.get(goal, 0) > 0:
                    meta_goal_stats[goal] += 1

    print(f"\nМЕТАЦЕЛИ:")
    for goal, count in meta_goal_stats.items():
        percentage = (count / final_count * 100) if final_count > 0 else 0
        print(f"  {goal}: {count} агентов ({percentage:.1f}%)")

    # Анализ отдельных агентов
    print(f"\nДЕТАЛЬНЫЙ АНАЛИЗ АГЕНТОВ:")
    for agent in sorted(agents, key=lambda x: x.id):
        if not agent.alive:
            continue

        print(f"\nАгент {agent.id}:")
        if hasattr(agent, 'parent_id') and agent.parent_id is not None:
            print(f"  Потомок агента {agent.parent_id}")
        if hasattr(agent, 'ancestors'):
            print(f"  Предки: {sorted(agent.ancestors)}")

        print(f"  Возраст: {agent.age} эпизодов")
        if hasattr(agent, 'activation_episodes'):
            print(f"  Активации метацелей: {agent.activation_episodes}")
        print(f"  Уровень: {getattr(agent, 'level', 1)}, Концептов: {getattr(agent, 'num_concepts', 0)}")
        print(
            f"  Ресурсы: S_self={getattr(agent, 'energy', 0):.1f}, S_pred={getattr(agent, 'knowledge', 0):.1f}, S_exp={getattr(agent, 'territory', 0):.1f}")
        print(f"  Показатели: P={getattr(agent, 'P', 0):.1f}, S={getattr(agent, 'S', 0):.1f}")

        # Кооперация
        if hasattr(agent, 'cooperation_history') and agent.cooperation_history:
            coop_count = sum(agent.cooperation_history.values())
            print(f"  Кооперация: {coop_count} взаимодействий")
            print(f"  Партнеры: {list(agent.cooperation_history.keys())}")

        # Достигнутые пороги
        thresholds_reached = []
        if hasattr(agent, 'thresholds'):
            final_C = agent.history['complexity'][-1] if agent.history.get('complexity') else 0
            final_Phi = agent.history['phi'][-1] if agent.history.get('phi') else 0
            final_E_eff = agent.history['e_eff'][-1] if agent.history.get('e_eff') else 0

            if final_C > agent.thresholds.get('C', 0):
                thresholds_reached.append('C')
            if final_Phi > agent.thresholds.get('Phi', 0):
                thresholds_reached.append('Φ')
            if final_E_eff > agent.thresholds.get('E_eff', 0):
                thresholds_reached.append('E_eff')

        print(f"  Достигнутые пороги: {thresholds_reached}")

        if agent.history.get('fractal_dim'):
            final_fractal_dim = agent.history['fractal_dim'][-1]
            print(f"  Фрактальная размерность: {final_fractal_dim:.2f}")

    # Генеалогическое древо
    genealogy = {}
    for agent in agents:
        if hasattr(agent, 'parent_id') and agent.parent_id is not None:
            if agent.parent_id not in genealogy:
                genealogy[agent.parent_id] = []
            genealogy[agent.parent_id].append(agent.id)

    if genealogy:
        print(f"\nГЕНЕАЛОГИЧЕСКОЕ ДРЕВО:")
        for parent, children in sorted(genealogy.items()):
            print(f"  Агент {parent} -> {children}")

    print(f"\nСИМУЛЯЦИЯ ЗАВЕРШЕНА. Всего эпизодов: {episodes}")