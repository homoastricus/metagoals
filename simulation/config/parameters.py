# Параметры обучения с подкреплением
RL_PARAMS = {
    'alpha': 0.05,  # Скорость обучения
    'gamma': 0.85,  # Коэффициент дисконтирования
    'base_exploration': 0.3,  # Базовая вероятность исследования
    'exploration_bonus_pred': 0.1,  # Бонус к исследованию при активации S_pred
    'action_bias_exp': 0.1,  # Смещение выбора действия при активации S_exp
}

# Параметры метацелей и их активации
META_GOAL_PARAMS = {
    'energy_base_cost': 0.02,  # Базовая стоимость существования
    'energy_reward_factor': 0.1,  # Коэффициент преобразования reward в энергию
    'energy_self_multiplier': 1.1,  # Множитель затрат энергии при активированной S_self
    'knowledge_learning_factor': 0.1,  # Коэффициент обучения для роста знания
    'knowledge_pred_multiplier': 1.1,  # Множитель обучения при активированной S_pred
    'territory_gain_base': 0.02,  # Базовый прирост территории
    'territory_exp_multiplier': 1.6,  # Множитель прироста территории при активированной S_exp
    'energy_threshold_self': 8.0,  # Порог энергии для активации S_self
    'territory_threshold_exp': 1.5,  # Порог территории для активации S_exp
    'threshold_scale_factor': 0.99,  # Масштабный фактор для порога S_self
    'entropy_barrier_self': 0.1,  # Энтропийный барьер для S_self
    'entropy_barrier_pred': 0.01,  # Энтропийный барьер для S_pred
    'entropy_barrier_exp': 0.15,  # Энтропийный барьер для S_exp
    'activation_entropy_cost_self': 0.05,  # Энтропийная стоимость активации S_self
    'activation_entropy_cost_pred': 0.03,  # Энтропийная стоимость активации S_pred
    'activation_entropy_cost_exp': 0.08,  # Энтропийная стоимость активации S_exp
    'entropy_dissipation_ratio': 0.7,  # Доля энтропии, которая диссипирует
    'meta_goal_competition_factor': 1.1,  # Фактор конкуренции между метацелями
    'max_knowledge': 500.0,  # Ограничиваем рост знаний
}

# Параметры Уравнения Динамики Порядка (УДП)
ORDER_DYNAMICS_PARAMS = {
    'alpha_transposition': 0.3,  # Коэффициент энтропической транспозиции
    'delta_decay': 0.1,  # Коэффициент распада порядка
    'xi_mean': 0.0,  # Среднее значение стохастического шума
    'xi_std': 0.1,  # Стандартное отклонение стохастического шума
    'theta_critical': 500.0,  # Базовый коэффициент для критического порога P_crit
    'beta_F': 1.9,  # Показатель степени для функции F(C) = C^beta
    'gamma_g': 1.2,  # Показатель степени для функции g(C) = C^gamma
    'zeta_h': 1.0,  # Показатель степени для функции h(S) = S^zeta
    'entropy_export_ratio': 0.3,  # Доля энтропии, сбрасываемой при переходе на новый уровень
    'entropy_keep_ratio': 0.7,  # Доля энтропии, остающейся после перехода на новый уровень
    'alpha_growth_factor': 1.2,  # Множитель роста alpha после перехода на новый уровень
}

# Параметры социальных взаимодействий
SOCIAL_PARAMS = {
    'social_influence_factor': 0.15,  # Сила влияния знаний других агентов
}

# Параметры истории и анализа
ANALYSIS_PARAMS = {
    'history_window_size': 50,  # Размер окна для расчета скользящих статистик
    'baseline_window_size': 200,  # Размер окна для расчета базовых значений порогов
    'min_samples_for_threshold': 20,  # Минимальное количество samples для расчета порогов
}

# Параметры среды
ENVIRONMENT_PARAMS = {
    'num_states': 30,  # Количество состояний среды
    'num_actions': 8,  # Количество возможных действий
    'agent_reward_variability': 0.3,  # Вариабельность наград между агентами
    'min_reward': 0.8,  # Уменьшаем минимальную награду
    'max_reward': 2.5,  # Уменьшаем максимальную награду
    'pattern_change_interval': 270,  # Интервал изменения паттернов среды
    'hostility_increase_rate': 0.001,  # Скорость увеличения враждебности среды
}

# Параметры агента
AGENT_PARAMS = {
    'initial_energy': 25.0,  # Начальный уровень энергии
    'max_energy': 88.0,  # Максимальный уровень энергии
    'initial_territory': 3.0,  # Начальный размер территории
    'learning_rate_variability': 0.2,  # Вариабельность скорости обучения
    'territory_gain_variability': 0.3,  # Вариабельность прироста территории
    'energy_efficiency_variability': 0.2,  # Вариабельность энергоэффективности
}

# Параметры размножения и смерти
LIFE_PARAMS = {
    'reproduction_energy_threshold': 8.2,  # Порог энергии для размножения
    'reproduction_territory_threshold': 6.2,  # Порог территории для размножения
    'reproduction_knowledge_threshold': 3.2,  # Порог знания для размножения
    'reproduction_social_influence_bonus': 1.0,  # Бонус к социальному влиянию при рождении нового агента
    'death_energy_threshold': 3.0,  # Порог энергии, ниже которого агент умирает
    'death_entropy_threshold': 3.9,  # Порог энтропии, выше которого агент умирает (если среда враждебная)
    'hostile_environment_factor': 0.000093,  # Фактор враждебности среды (увеличивает смертность)
    'max_age': 2900,  # Максимальный возраст агента
    'age_mortality_factor': 0.00020,  # Фактор смертности с возрастом
}

# Новые параметры для мутаций и кооперации
EVOLUTION_PARAMS = {
    'mutation_rate': 0.15,  # Вероятность мутации параметров
    'mutation_strength': 0.1,  # Сила мутации
    'cooperation_benefit': 0.6,  # Бонус от кооперации
    'cooperation_cost': 0.05,  # Стоимость кооперации
    'kinship_recognition': 0.7,  # Вероятность распознавания родства
}