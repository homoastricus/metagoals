from collections import deque

import networkx as nx
import numpy as np

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

from ..config.parameters import ENVIRONMENT_PARAMS
from ..config.parameters import AGENT_PARAMS
from ..config.parameters import ANALYSIS_PARAMS
from ..config.parameters import ORDER_DYNAMICS_PARAMS
from ..config.parameters import RL_PARAMS
from ..config.parameters import SOCIAL_PARAMS
from ..config.parameters import META_GOAL_PARAMS
from ..config.parameters import LIFE_PARAMS
from ..config.parameters import EVOLUTION_PARAMS


class AdaptiveAgent:
    def __init__(self, num_states=ENVIRONMENT_PARAMS['num_states'],
                 num_actions=ENVIRONMENT_PARAMS['num_actions'], agent_id=0, parent_agent=None):
        self.id = agent_id
        self.num_states = num_states
        self.num_actions = num_actions

        # Наследование или новая инициализация
        if parent_agent:
            self._inherit_from_parent(parent_agent)
            self._apply_mutations()
        else:
            self._initialize_new_agent()

        # Новые атрибуты для эволюции
        self.age = 0
        self.alive = True
        self.parent_id = parent_agent.id if parent_agent else None
        self.ancestors = set()
        if parent_agent:
            self.ancestors = parent_agent.ancestors.copy()
            self.ancestors.add(parent_agent.id)

        self.cooperation_history = {}
        self.birth_episode = 0

        # Инициализация Q-таблицы
        if not hasattr(self, 'Q'):
            self.Q = np.random.rand(num_states, num_actions) * 0.1 + 0.01

        # Метацели с адаптивными порогами
        if not hasattr(self, 'energy'):
            self.energy = AGENT_PARAMS['initial_energy']
        if not hasattr(self, 'knowledge'):
            self.knowledge = 0.0
        if not hasattr(self, 'territory'):
            self.territory = AGENT_PARAMS['initial_territory']

        # Флаги активации метацелей
        self.meta_active = {'self': False, 'pred': False, 'exp': False}
        self.activation_episodes = {'self': 0, 'pred': 0, 'exp': 0}
        self.meta_deactivation_timer = {'self': 0, 'pred': 0, 'exp': 0}

        # Адаптивные пороги
        self.thresholds = {'C': 0, 'Phi': 0, 'E_eff': 0, 'S': 0}
        self.baseline_values = {
            'C': deque(maxlen=ANALYSIS_PARAMS['baseline_window_size']),
            'Phi': deque(maxlen=ANALYSIS_PARAMS['baseline_window_size']),
            'E_eff': deque(maxlen=ANALYSIS_PARAMS['baseline_window_size']),
            'S': deque(maxlen=ANALYSIS_PARAMS['baseline_window_size'])
        }

        # Уровни эмерджентности
        self.level = 1
        self.P = 0.0
        self.S = 0.0
        self.alpha = ORDER_DYNAMICS_PARAMS['alpha_transposition']
        self.delta = ORDER_DYNAMICS_PARAMS['delta_decay']
        self.Xi_mean = ORDER_DYNAMICS_PARAMS['xi_mean']
        self.Xi_std = ORDER_DYNAMICS_PARAMS['xi_std']
        self.theta = ORDER_DYNAMICS_PARAMS['theta_critical']
        self.beta_F = ORDER_DYNAMICS_PARAMS['beta_F']
        self.P_crit = self.theta * (1 ** self.beta_F)
        self.level_up_episode = None
        self.gamma_g = ORDER_DYNAMICS_PARAMS['gamma_g']
        self.zeta_h = ORDER_DYNAMICS_PARAMS['zeta_h']
        self.S_lower = 0.0
        self.entropy_export_ratio = ORDER_DYNAMICS_PARAMS['entropy_export_ratio']
        self.entropy_keep_ratio = ORDER_DYNAMICS_PARAMS['entropy_keep_ratio']
        self.alpha_growth_factor = ORDER_DYNAMICS_PARAMS['alpha_growth_factor']

        # Индивидуальные параметры агента
        if not hasattr(self, 'individual_learning_rate'):
            self.individual_learning_rate = RL_PARAMS['alpha'] * np.random.lognormal(0, AGENT_PARAMS[
                'learning_rate_variability'])
        if not hasattr(self, 'individual_territory_gain'):
            self.individual_territory_gain = META_GOAL_PARAMS['territory_gain_base'] * np.random.lognormal(0,
                                                                                                           AGENT_PARAMS[
                                                                                                               'territory_gain_variability'])
        if not hasattr(self, 'individual_energy_efficiency'):
            self.individual_energy_efficiency = np.random.lognormal(0, AGENT_PARAMS['energy_efficiency_variability'])

        # Индивидуальные энтропийные барьеры
        self.entropy_barriers = {
            'self': META_GOAL_PARAMS['entropy_barrier_self'] * np.random.lognormal(0, 0.2),
            'pred': META_GOAL_PARAMS['entropy_barrier_pred'] * np.random.lognormal(0, 0.2),
            'exp': META_GOAL_PARAMS['entropy_barrier_exp'] * np.random.lognormal(0, 0.2)
        }

        # Индивидуальная стоимость активации
        self.activation_costs = {
            'self': META_GOAL_PARAMS['activation_entropy_cost_self'] * np.random.lognormal(0, 0.15),
            'pred': META_GOAL_PARAMS['activation_entropy_cost_pred'] * np.random.lognormal(0, 0.15),
            'exp': META_GOAL_PARAMS['activation_entropy_cost_exp'] * np.random.lognormal(0, 0.15)
        }

        # История для анализа
        self.history = {
            'energy': [], 'knowledge': [], 'territory': [],
            'complexity': [], 'phi': [], 'e_eff': [],
            'states': [], 'actions': [], 'rewards': [],
            'exploration_rate': [], 'meta_active': [],
            'P': [], 'S': [], 'level': [], 'fractal_dim': [],
            'entropy_cost': [], 'effective_thresholds': [],
            'cooperation_benefit': [], 'age': []
        }

        # Для визуализации состояний
        self.state_transitions = np.zeros((num_states, num_states)) + 1e-10
        self.last_state = None
        self.G = nx.DiGraph()

        # Параметры поведения
        self.base_exploration = RL_PARAMS['base_exploration']
        self.exploration_rate = self.base_exploration
        self.exploration_bonus_pred = RL_PARAMS['exploration_bonus_pred']
        self.action_bias_exp = RL_PARAMS['action_bias_exp']

        # Индивидуальные reward множители
        if not hasattr(self, 'reward_multiplier'):
            self.reward_multiplier = np.random.lognormal(0, ENVIRONMENT_PARAMS['agent_reward_variability'])

        # Новая архитектура для уровней > 1
        self.concept_map = None
        self.num_concepts = 0
        self.Q_concepts = None
        self.current_concept = None

        # Инициализация знания с ограничением и безопасными значениями
        if parent_agent:
            self.knowledge = parent_agent.knowledge * 0.8
        else:
            self.knowledge = 0.0

        # Применяем ограничение с значениями по умолчанию
        max_knowledge = META_GOAL_PARAMS.get('max_knowledge', 1000.0) if hasattr(META_GOAL_PARAMS, 'get') else 1000.0
        min_knowledge = META_GOAL_PARAMS.get('min_knowledge', 0.1) if hasattr(META_GOAL_PARAMS, 'get') else 0.1
        self.knowledge = min(max(self.knowledge, min_knowledge), max_knowledge)

    def _inherit_from_parent(self, parent):
        """Наследование параметров от родителя"""
        self.energy = parent.energy * 0.7
        self.knowledge = parent.knowledge * 0.8
        self.territory = parent.territory * 0.6

        # Наследование Q-таблицы
        self.Q = parent.Q.copy() * np.random.normal(1.0, 0.05, parent.Q.shape)

        # Наследование индивидуальных параметров
        self.individual_learning_rate = parent.individual_learning_rate
        self.individual_territory_gain = parent.individual_territory_gain
        self.individual_energy_efficiency = parent.individual_energy_efficiency
        self.reward_multiplier = parent.reward_multiplier

    def _apply_mutations(self):
        """Применение мутаций к параметрам агента"""
        if np.random.random() < EVOLUTION_PARAMS['mutation_rate']:
            self.individual_learning_rate *= np.random.normal(1.0, EVOLUTION_PARAMS['mutation_strength'])

        if np.random.random() < EVOLUTION_PARAMS['mutation_rate']:
            self.individual_territory_gain *= np.random.normal(1.0, EVOLUTION_PARAMS['mutation_strength'])

        if np.random.random() < EVOLUTION_PARAMS['mutation_rate']:
            self.individual_energy_efficiency *= np.random.normal(1.0, EVOLUTION_PARAMS['mutation_strength'])

        if np.random.random() < EVOLUTION_PARAMS['mutation_rate']:
            self.reward_multiplier *= np.random.normal(1.0, EVOLUTION_PARAMS['mutation_strength'])

    def _initialize_new_agent(self):
        """Инициализация нового агента без родителя"""
        self.energy = AGENT_PARAMS['initial_energy']
        self.knowledge = 0.0
        self.territory = AGENT_PARAMS['initial_territory']
        self.Q = np.random.rand(self.num_states, self.num_actions) * 0.1 + 0.01
        self.individual_learning_rate = RL_PARAMS['alpha'] * np.random.lognormal(0, AGENT_PARAMS[
            'learning_rate_variability'])
        self.individual_territory_gain = META_GOAL_PARAMS['territory_gain_base'] * np.random.lognormal(0, AGENT_PARAMS[
            'territory_gain_variability'])
        self.individual_energy_efficiency = np.random.lognormal(0, AGENT_PARAMS['energy_efficiency_variability'])
        self.reward_multiplier = np.random.lognormal(0, ENVIRONMENT_PARAMS['agent_reward_variability'])

    def _compute_phi(self):
        """Вычисляет интегрированную информацию (Φ)"""
        if np.sum(self.state_transitions) < 1e-10:
            return 0.0

        try:
            tpm = self.state_transitions.astype(float)
            row_sums = tpm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            tpm_normalized = tpm / row_sums
            tpm_normalized = np.nan_to_num(tpm_normalized, nan=0.0, posinf=0.0, neginf=0.0)
            phi_val = np.trace(tpm_normalized) / self.num_states
            return phi_val
        except:
            return 0.0

    def _compute_fractal_dimension_simple(self):
        """Упрощённый расчёт фрактальной размерности"""
        if len(self.G.nodes) < 2:
            return 0.0
        try:
            num_nodes = len(self.G.nodes)
            num_edges = len(self.G.edges)
            if num_edges == 0 or num_nodes == 0:
                return 0.0
            fractal_dim = np.log(num_edges + 1) / np.log(num_nodes + 1)
            return min(fractal_dim, 2.0)
        except:
            return 0.0

    def _form_concepts(self):
        """Группирует состояния в концепты на основе их Q-ценности."""
        if KMeans is None or len(self.Q) < 2:
            self.concept_map = {0: list(range(self.num_states))}
            self.num_concepts = 1
            return

        try:
            self.num_concepts = max(2, min(int(np.sqrt(self.num_states)), self.num_states))
            Q_normalized = (self.Q - np.mean(self.Q, axis=0)) / (np.std(self.Q, axis=0) + 1e-10)
            Q_normalized = np.nan_to_num(Q_normalized, nan=0.0)
            kmeans = KMeans(n_clusters=self.num_concepts, random_state=0, n_init=10, max_iter=100)
            labels = kmeans.fit_predict(Q_normalized)
            self.concept_map = {}
            for state, concept_id in enumerate(labels):
                if concept_id not in self.concept_map:
                    self.concept_map[concept_id] = []
                self.concept_map[concept_id].append(state)
            self.current_concept = None
        except:
            self.concept_map = {0: list(range(self.num_states))}
            self.num_concepts = 1

    def _transfer_knowledge_to_new_level(self):
        """Переносит знания из Q-таблицы старых состояний в новую таблицу концептов."""
        if self.concept_map is None:
            return
        self.Q_concepts = np.zeros((self.num_concepts, self.num_actions))
        for concept_id, state_list in self.concept_map.items():
            if state_list:
                self.Q_concepts[concept_id] = np.mean(self.Q[state_list, :], axis=0)

    def choose_action(self, state, other_agents=None):
        if self.level > 1 and self.concept_map is not None and self.Q_concepts is not None:
            concept_id = None
            for c_id, state_list in self.concept_map.items():
                if state in state_list:
                    concept_id = c_id
                    break
            if concept_id is None:
                concept_id = 0
            self.current_concept = concept_id
            social_influence = 0
            if other_agents:
                for other in other_agents:
                    if other.id != self.id:
                        social_influence += other.knowledge * SOCIAL_PARAMS['social_influence_factor']
            exploration_bonus = self.exploration_bonus_pred if self.meta_active['pred'] else 0
            action_bias = self.action_bias_exp if self.meta_active['exp'] else 0
            current_exploration = self.exploration_rate + exploration_bonus
            if np.random.random() < current_exploration:
                action = np.random.randint(self.num_actions)
            else:
                q_values = self.Q_concepts[concept_id] + social_influence + action_bias
                action = np.argmax(q_values)
        else:
            social_influence = 0
            if other_agents:
                for other in other_agents:
                    if other.id != self.id:
                        social_influence += other.knowledge * SOCIAL_PARAMS['social_influence_factor']
            exploration_bonus = self.exploration_bonus_pred if self.meta_active['pred'] else 0
            action_bias = self.action_bias_exp if self.meta_active['exp'] else 0
            current_exploration = self.exploration_rate + exploration_bonus
            if np.random.random() < current_exploration:
                action = np.random.randint(self.num_actions)
            else:
                q_values = self.Q[state] + social_influence + action_bias
                action = np.argmax(q_values)
        self.history['exploration_rate'].append(current_exploration)
        return action

    def _safe_entropy(self, probs):
        """Безопасное вычисление энтропии"""
        try:
            probs = np.asarray(probs)
            if np.any(probs < 0) or np.abs(np.sum(probs) - 1.0) > 0.01:
                probs = np.clip(probs, 1e-10, 1.0)
                probs = probs / np.sum(probs)
            log_probs = np.log2(probs + 1e-10)
            entropy_val = -np.sum(probs * log_probs)
            return max(0.0, entropy_val)
        except:
            return 0.0

    def update(self, state, action, reward, next_state, other_agents=None):
        # Увеличиваем возраст
        self.age += 1
        self.history['age'].append(self.age)

        adjusted_reward = reward * self.reward_multiplier
        self.history['states'].append(state)
        self.history['actions'].append(action)
        self.history['rewards'].append(adjusted_reward)

        # Обновляем граф переходов
        if self.last_state is not None:
            self.state_transitions[self.last_state, state] += 1
            self.G.add_edge(self.last_state, state, weight=self.state_transitions[self.last_state, state])
        self.last_state = state

        # Q-learning с адаптивным обучением
        old_value = self.Q[state, action]
        next_max = np.max(self.Q[next_state])
        new_value = old_value + self.individual_learning_rate * (
                adjusted_reward + RL_PARAMS['gamma'] * next_max - old_value)
        self.Q[state, action] = new_value

        # Обновление метацелей
        energy_base_change = adjusted_reward * META_GOAL_PARAMS[
            'energy_reward_factor'] * self.individual_energy_efficiency - META_GOAL_PARAMS['energy_base_cost']
        if self.meta_active['self']:
            energy_change = energy_base_change * META_GOAL_PARAMS['energy_self_multiplier']
        else:
            energy_change = energy_base_change
        self.energy = max(0, min(AGENT_PARAMS['max_energy'], self.energy + energy_change))

        knowledge_base_gain = abs(new_value - old_value) * META_GOAL_PARAMS.get('knowledge_learning_factor', 0.2)
        if self.meta_active['pred']:
            knowledge_gain = knowledge_base_gain * META_GOAL_PARAMS.get('knowledge_pred_multiplier', 1.4)
        else:
            knowledge_gain = knowledge_base_gain
        self.knowledge += knowledge_gain

        # ОГРАНИЧЕНИЕ ЗНАНИЯ с безопасными значениями по умолчанию
        max_knowledge = META_GOAL_PARAMS.get('max_knowledge', 1000.0)
        min_knowledge = META_GOAL_PARAMS.get('min_knowledge', 0.1)
        self.knowledge = min(max(self.knowledge, min_knowledge), max_knowledge)

        territory_base_gain = self.individual_territory_gain
        if self.meta_active['exp']:
            territory_gain = territory_base_gain * META_GOAL_PARAMS['territory_exp_multiplier']
        else:
            territory_gain = territory_base_gain
        if adjusted_reward > 0:
            self.territory += territory_gain

        # Сохраняем значения для адаптивных порогов
        recent_states = self.history['states'][-ANALYSIS_PARAMS['history_window_size']:] if len(
            self.history['states']) > ANALYSIS_PARAMS['history_window_size'] else self.history['states']
        if recent_states:
            state_counts = np.bincount(recent_states, minlength=self.num_states)
            state_probs = state_counts.astype(float) / len(recent_states)
            complexity = self._safe_entropy(state_probs)
        else:
            complexity = 0

        phi_val = self._compute_phi()
        e_eff = 0.3 * self.energy + 0.4 * self.knowledge + 0.3 * self.territory

        self.baseline_values['C'].append(complexity)
        self.baseline_values['Phi'].append(phi_val)
        self.baseline_values['E_eff'].append(e_eff)

        # Вычисляем адаптивные пороги
        if len(self.baseline_values['C']) > ANALYSIS_PARAMS['min_samples_for_threshold']:
            for key in ['C', 'Phi', 'E_eff']:
                values = list(self.baseline_values[key])
                if values:
                    self.thresholds[key] = np.mean(values) + np.std(values)

        # УДП: Обновление порядка P и энтропии S
        g_C = complexity ** self.gamma_g if complexity > 0 else 0
        h_S = self.S ** self.zeta_h if self.S > 0 else 0
        Xi = np.random.normal(self.Xi_mean, self.Xi_std)
        dP_dt = e_eff * g_C - self.alpha * h_S - self.delta * self.P + Xi
        self.P = max(0, self.P + dP_dt)

        # Обновление энтропии S
        if recent_states:
            state_counts = np.bincount(recent_states, minlength=self.num_states)
            state_probs = state_counts.astype(float) / len(recent_states)
            self.S = self._safe_entropy(state_probs)
        else:
            self.S = 0

        # Конкуренция метацелей
        for goal in self.meta_deactivation_timer:
            if self.meta_deactivation_timer[goal] > 0:
                self.meta_deactivation_timer[goal] -= 1
                if self.meta_deactivation_timer[goal] == 0:
                    self.meta_active[goal] = False

        # Сохраняем историю
        self.history['energy'].append(self.energy)
        self.history['knowledge'].append(self.knowledge)
        self.history['territory'].append(self.territory)
        self.history['complexity'].append(complexity)
        self.history['phi'].append(phi_val)
        self.history['e_eff'].append(e_eff)
        self.history['P'].append(self.P)
        self.history['S'].append(self.S)
        self.history['level'].append(self.level)
        fractal_dim = self._compute_fractal_dimension_simple()
        self.history['fractal_dim'].append(fractal_dim)

        return new_value - old_value

    def cooperate(self, other_agent):
        """Кооперативное взаимодействие с другим агентом"""
        if self.id == other_agent.id:
            return 0

        # Всегда инициализируем переменную cooperation_chance
        cooperation_chance = 0.3  # Значение по умолчанию для неродственников

        # Проверяем родство
        is_relative = False
        if hasattr(self, 'ancestors') and hasattr(other_agent, 'id'):
            is_relative = other_agent.id in self.ancestors or self.id in getattr(other_agent, 'ancestors', set())

        if is_relative:
            cooperation_chance = EVOLUTION_PARAMS.get('kinship_recognition', 0.7)

        if np.random.random() < cooperation_chance:
            # Обмен знаниями с ограничением
            knowledge_transfer = min(self.knowledge * 0.1, other_agent.knowledge * 0.1)

            # Применяем ограничение после передачи знаний
            self.knowledge += knowledge_transfer
            other_agent.knowledge += knowledge_transfer

            # Ограничение максимального знания
            max_knowledge = META_GOAL_PARAMS.get('max_knowledge', 1000.0)
            min_knowledge = META_GOAL_PARAMS.get('min_knowledge', 0.1)

            self.knowledge = min(max(self.knowledge, min_knowledge), max_knowledge)
            other_agent.knowledge = min(max(other_agent.knowledge, min_knowledge), max_knowledge)

            # Затраты на кооперацию
            cost = EVOLUTION_PARAMS.get('cooperation_cost', 0.1)
            self.energy -= cost
            other_agent.energy -= cost

            # Записываем взаимодействие
            if not hasattr(self, 'cooperation_history'):
                self.cooperation_history = {}
            if not hasattr(other_agent, 'cooperation_history'):
                other_agent.cooperation_history = {}

            self.cooperation_history[other_agent.id] = self.cooperation_history.get(other_agent.id, 0) + 1
            other_agent.cooperation_history[self.id] = other_agent.cooperation_history.get(self.id, 0) + 1

            return knowledge_transfer

        return 0

    def activate_meta_goal(self, goal_type, episode):
        """Активация метацели с учетом энтропийной стоимости"""
        activation_cost = self.activation_costs[goal_type] * self.S
        if self.S >= activation_cost:
            self.S_lower += activation_cost * META_GOAL_PARAMS['entropy_dissipation_ratio']
            self.S = max(0, self.S - activation_cost)
            self.meta_active[goal_type] = True
            self.activation_episodes[goal_type] = episode
            for other_goal in ['self', 'pred', 'exp']:
                if other_goal != goal_type and self.meta_active[other_goal]:
                    if np.random.random() < 0.7:
                        self.meta_deactivation_timer[other_goal] = np.random.randint(30, 70)
            self.history['entropy_cost'].append(activation_cost)
            return True
        return False

    def check_meta_goals(self, episode):

        """Проверка и активация метацелей с учетом энтропийных барьеров"""
        # ДОБАВЛЯЕМ проверку на минимальную энтропию
        if self.S < 0.1:  # Не позволяем активировать при нулевой энтропии
            return []
        """Проверка и активация метацелей с учетом энтропийных барьеров"""
        current_values = {
            'C': self.history['complexity'][-1] if self.history['complexity'] else 0,
            'Phi': self.history['phi'][-1] if self.history['phi'] else 0,
            'E_eff': self.history['e_eff'][-1] if self.history['e_eff'] else 0
        }

        new_activations = []
        effective_thresholds = {}

        effective_thresholds['E_eff_self'] = (
                self.thresholds['E_eff'] * META_GOAL_PARAMS['threshold_scale_factor'] + self.entropy_barriers[
            'self'] * self.S)
        effective_thresholds['E_eff_exp'] = (self.thresholds['E_eff'] + self.entropy_barriers['exp'] * self.S)
        effective_thresholds['C_pred'] = (self.thresholds['C'] + self.entropy_barriers['pred'] * self.S)
        effective_thresholds['Phi_pred'] = (self.thresholds['Phi'] + self.entropy_barriers['pred'] * self.S)

        # S_self
        if (current_values['E_eff'] > effective_thresholds['E_eff_self'] and
                self.energy > META_GOAL_PARAMS['energy_threshold_self'] and
                not self.meta_active['self'] and self.meta_deactivation_timer['self'] == 0):
            if self.activate_meta_goal('self', episode):
                new_activations.append('self')

        # S_pred
        if (current_values['C'] > effective_thresholds['C_pred'] and
                current_values['Phi'] > effective_thresholds['Phi_pred'] and
                not self.meta_active['pred'] and self.meta_deactivation_timer['pred'] == 0):
            if self.activate_meta_goal('pred', episode):
                new_activations.append('pred')

        # S_exp
        if (current_values['E_eff'] > effective_thresholds['E_eff_exp'] and
                self.territory > META_GOAL_PARAMS['territory_threshold_exp'] and
                not self.meta_active['exp'] and self.meta_deactivation_timer['exp'] == 0):
            if self.activate_meta_goal('exp', episode):
                new_activations.append('exp')

        # Проверка на рождение нового уровня
        if self.P > self.P_crit and self.level_up_episode is None:
            self.level += 1
            self.level_up_episode = episode
            self._form_concepts()
            self._transfer_knowledge_to_new_level()
            self.S_lower += self.S * self.entropy_export_ratio
            self.S *= self.entropy_keep_ratio
            self.P -= self.P_crit
            C_next = self.level * current_values['C']
            F_C_next = C_next ** self.beta_F
            self.P_crit = self.theta * F_C_next
            self.alpha *= self.alpha_growth_factor
            print(f"Агент {self.id}: Рождение уровня {self.level} на эпизоде {episode}! Концептов: {self.num_concepts}")

        self.history['meta_active'].append(self.meta_active.copy())
        self.history['effective_thresholds'].append(effective_thresholds.copy())

        return new_activations

    def check_reproduction_conditions(self):
        """Проверяет, выполнились ли условия для размножения"""
        return (self.energy > LIFE_PARAMS['reproduction_energy_threshold'] and
                self.territory > LIFE_PARAMS['reproduction_territory_threshold'] and
                self.knowledge > LIFE_PARAMS['reproduction_knowledge_threshold'])

    def reproduce(self, new_agent_id):
        """Создает нового агента с уменьшенными затратами для родителя"""
        new_agent = AdaptiveAgent(
            num_states=self.num_states,
            num_actions=self.num_actions,
            agent_id=new_agent_id,
            parent_agent=self
        )

        # Меньшие затраты для родителя
        self.energy *= 0.8
        self.territory *= 0.7

        return new_agent

    def check_death_conditions(self):
        """Проверяет, должен ли агент умереть с учетом возраста"""
        # Смерть от недостатка энергии
        if self.energy < LIFE_PARAMS['death_energy_threshold']:
            return True

        # Смерть от высокой энтропии
        if (self.S > LIFE_PARAMS['death_entropy_threshold'] and
                np.random.random() < LIFE_PARAMS['hostile_environment_factor']):
            return True

        # Смерть от старости
        age_mortality = min(0.9, self.age * LIFE_PARAMS['age_mortality_factor'])
        if np.random.random() < age_mortality:
            return True

        return False
