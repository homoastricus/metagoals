import numpy as np
from .environment import Environment


class ComplexEnvironment(Environment):
    """
    Новая, более сложная среда для проверки гипотез о третьем уровне эмерджентности.
    Вводит глобальные, вычислимые закономерности, которые агенты могут пытаться предсказать.
    """

    def __init__(self, num_states=50, num_actions=10):
        super().__init__(num_states)

        # Параметры глобальных динамических процессов
        self.global_phase = 0.0
        self.phase_increment = 0.01

        # 1. "Сезонные" циклы
        self.seasonal_amplitude = 0.5
        self.seasonal_period = 500

        # 2. "Тренды"
        self.current_trend = 0.0
        self.trend_direction = 1
        self.trend_change_prob = 0.002

        # 3. "Скрытые режимы"
        self.hidden_modes = ['calm', 'volatile', 'generous']
        self.current_mode = 'calm'
        self.mode_duration = 0
        self.max_mode_duration = 200

        # 4. Долгосрочные циклы
        self.long_cycle_period = 2000
        self.long_cycle_phase = 0

    def get_global_dynamics_factor(self, episode):
        """Вычисляет комплексный фактор, влияющий на ВСЕ награды в среде."""
        # 1. Сезонность
        seasonal = np.sin(2 * np.pi * episode / self.seasonal_period)

        # 2. Тренд
        trend_effect = self.current_trend

        # 3. Режим
        mode_effect = 0.0
        if self.current_mode == 'volatile':
            mode_effect = np.random.uniform(-1.0, 1.0)
        elif self.current_mode == 'generous':
            mode_effect = 0.5

        # 4. Долгосрочный цикл
        long_cycle = 0.3 * np.sin(2 * np.pi * episode / self.long_cycle_period)

        # Композитный фактор динамики
        dynamics_factor = (self.seasonal_amplitude * seasonal +
                           trend_effect +
                           mode_effect +
                           long_cycle)

        return dynamics_factor

    def update_global_dynamics(self, episode):
        """Обновляет внутреннее состояние глобальной динамики среды."""
        # Обновляем тренд
        if np.random.random() < self.trend_change_prob:
            self.trend_direction *= -1
        self.current_trend += 0.001 * self.trend_direction
        self.current_trend = np.clip(self.current_trend, -0.5, 0.5)

        # Обновляем скрытый режим
        self.mode_duration += 1
        if self.mode_duration > self.max_mode_duration:
            self.current_mode = np.random.choice(self.hidden_modes)
            self.mode_duration = 0

        # Обновляем долгосрочный цикл
        self.long_cycle_phase += 0.001
        if self.long_cycle_phase > 2 * np.pi:
            self.long_cycle_phase = 0

    def get_reward(self, state, action, episode, agent_id=None):
        # Базовая награда
        base_reward = super().get_reward(state, action, episode, agent_id)

        # Глобальный динамический фактор
        global_factor = self.get_global_dynamics_factor(episode)

        # Обновляем внутреннюю динамику
        self.update_global_dynamics(episode)

        # Модулируем награду
        modulated_reward = base_reward * (1 + global_factor)

        # Штрафы/бонусы в зависимости от режима
        if self.current_mode == 'volatile' and action > 7:
            modulated_reward *= 0.5
        elif self.current_mode == 'generous' and action < 2:
            modulated_reward *= 0.8

        return modulated_reward

    def get_observable_context(self, episode):
        """Возвращает наблюдаемые признаки глобальной динамики."""
        context = np.array([
            episode / 1000.0,
            np.sin(2 * np.pi * episode / 100),
            self.current_trend * 2,
            np.sin(self.long_cycle_phase),
            1.0 if self.current_mode == 'volatile' else 0.0,
            1.0 if self.current_mode == 'generous' else 0.0
        ])
        return context

    def get_environment_info(self):
        """Возвращает информацию о текущем состоянии среды для анализа."""
        return {
            'current_mode': self.current_mode,
            'current_trend': self.current_trend,
            'mode_duration': self.mode_duration
        }