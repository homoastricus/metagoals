import numpy as np
from simulation.config.parameters import ENVIRONMENT_PARAMS


class Environment:
    def __init__(self, num_states=ENVIRONMENT_PARAMS['num_states']):
        self.num_states = num_states
        self.num_actions = ENVIRONMENT_PARAMS['num_actions']

        # Динамическое создание базовых паттернов для любого количества состояний
        self.base_patterns = self._generate_base_patterns(num_states)
        self.patterns = self.base_patterns.copy()
        self.change_pattern_every = ENVIRONMENT_PARAMS['pattern_change_interval']
        self.agent_pattern_variations = {}  # Индивидуальные вариации для агентов
        self.hostility = 1.0  # Уровень враждебности среды
        self.hostility_increase_rate = ENVIRONMENT_PARAMS['hostility_increase_rate']

    def _generate_base_patterns(self, num_states):
        """Генерирует паттерны для произвольного количества состояний"""
        patterns = {}
        for state in range(num_states):
            # Создаем детерминированные, но разнообразные паттерны
            optimal_action = state % self.num_actions
            base_reward = 0.5 + (state * 0.3) % 1.5  # Вариация наград от 0.5 до 2.0
            patterns[state] = (optimal_action, base_reward)
        return patterns

    def get_reward(self, state, action, episode, agent_id=None):
        # Увеличиваем враждебность среды со временем
        self.hostility += self.hostility_increase_rate

        if episode % self.change_pattern_every == 0:
            self.shuffle_patterns()

        # Проверка на корректность состояния
        if state not in self.patterns:
            optimal_action = state % self.num_actions
            max_reward = ENVIRONMENT_PARAMS['min_reward']
        else:
            optimal_action, max_reward = self.patterns[state]

        # Добавляем индивидуальную вариабельность для агентов
        if agent_id is not None:
            if agent_id not in self.agent_pattern_variations:
                variation = np.random.normal(1.0, 0.15)
                self.agent_pattern_variations[agent_id] = variation
            max_reward *= self.agent_pattern_variations[agent_id]

        # Учитываем враждебность среды
        max_reward /= self.hostility

        similarity = 1.0 - abs(action - optimal_action) / (self.num_actions - 1)
        return max(ENVIRONMENT_PARAMS['min_reward'],
                   min(ENVIRONMENT_PARAMS['max_reward'], max_reward * similarity))

    def shuffle_patterns(self):
        """Обновляет паттерны, добавляя случайные вариации"""
        for state in range(self.num_states):  # Используем range для всех состояний
            if state in self.base_patterns:
                base_action, base_reward = self.base_patterns[state]
            else:
                # Паттерн по умолчанию для новых состояний
                base_action = state % self.num_actions
                base_reward = 0.5 + (state * 0.3) % 1.5

            # Небольшие случайные изменения
            new_action = (base_action + np.random.randint(-1, 2)) % self.num_actions
            new_reward = base_reward * np.random.uniform(0.8, 1.2)
            new_reward = max(ENVIRONMENT_PARAMS['min_reward'],
                             min(ENVIRONMENT_PARAMS['max_reward'], new_reward))
            self.patterns[state] = (new_action, new_reward)