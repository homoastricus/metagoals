import numpy as np
from collections import deque
import networkx as nx
from sklearn.linear_model import LinearRegression
from .adaptive_agent import AdaptiveAgent


class AdvancedAgent(AdaptiveAgent):
    def __init__(self, num_states=50, num_actions=10, agent_id=0):
        super().__init__(num_states, num_actions, agent_id)

        # Увеличиваем сложность среды
        self.num_states = num_states
        self.num_actions = num_actions

        # Увеличиваем пороги для более сложной среды
        self.thresholds = {
            'C': 2.0, 'Phi': 0.3, 'E_eff': 25.0, 'S': 1.5
        }

        # Увеличиваем максимальную энергию
        self.max_energy = 100.0

        # Уровень 3 атрибуты
        self.P_crit_level_3 = 5000.0
        self.trend_predictor = LinearRegression()
        self.context_memory = deque(maxlen=100)
        self.reward_memory = deque(maxlen=100)
        self.knowledge_graph = nx.DiGraph()
        self.meta_rules = []
        self.rule_confidences = {}

        # История для анализа уровня 3
        self.history['meta_rules_count'] = []
        self.history['knowledge_graph_complexity'] = []
        self.history['prediction_accuracy'] = []

    def perceive_environment_context(self, env, episode, current_reward):
        """Воспринимает контекст среды и обновляет предсказатель."""
        if hasattr(env, 'get_observable_context'):
            context = env.get_observable_context(episode)
            self.context_memory.append(context)
            self.reward_memory.append(current_reward)

            # Обучаем предсказатель тренда на последних данных
            if len(self.context_memory) > 10:
                X = list(self.context_memory)
                y = list(self.reward_memory)
                try:
                    self.trend_predictor.fit(X, y)
                    # Сохраняем точность предсказания
                    prediction = self.trend_predictor.predict([context])[0]
                    accuracy = 1.0 - abs(prediction - current_reward) / (abs(current_reward) + 1e-10)
                    self.history['prediction_accuracy'].append(accuracy)
                except:
                    pass

    def build_knowledge_graph(self):
        """Строит граф знаний на основе переходов между концептами."""
        if not hasattr(self, 'concept_map') or self.concept_map is None:
            return

        for i in range(len(self.history['states']) - 1):
            state_from = self.history['states'][i]
            state_to = self.history['states'][i + 1]

            concept_from = self.map_state_to_concept(state_from)
            concept_to = self.map_state_to_concept(state_to)

            if concept_from is not None and concept_to is not None:
                if self.knowledge_graph.has_edge(concept_from, concept_to):
                    self.knowledge_graph[concept_from][concept_to]['weight'] += 1
                else:
                    self.knowledge_graph.add_edge(concept_from, concept_to, weight=1,
                                                  rewards=[], count=1)

    def infer_meta_rules(self):
        """Выводит мета-правила из графа знаний."""
        if self.knowledge_graph.number_of_nodes() == 0:
            return

        new_rules = []

        for concept_from in self.knowledge_graph.nodes:
            successors = list(self.knowledge_graph.successors(concept_from))
            if successors:
                # Находим наиболее вероятные переходы
                best_next = max(successors, key=lambda x: self.knowledge_graph[concept_from][x]['weight'])
                weight = self.knowledge_graph[concept_from][best_next]['weight']

                # Проверяем качество перехода (средняя награда)
                if weight > 3:  # Минимальное количество наблюдений
                    rule_id = f"{concept_from}->{best_next}"

                    if rule_id not in self.rule_confidences:
                        self.rule_confidences[rule_id] = {
                            'count': 0,
                            'total_reward': 0.0,
                            'last_applied': 0
                        }

                    rule = {
                        'id': rule_id,
                        'condition': concept_from,
                        'predicted_outcome': best_next,
                        'confidence': weight,
                        'quality': 0.0  # Будет вычислено ниже
                    }
                    new_rules.append(rule)

        # Обновляем качество правил
        self.meta_rules = new_rules
        self.history['meta_rules_count'].append(len(self.meta_rules))
        self.history['knowledge_graph_complexity'].append(
            self.knowledge_graph.number_of_edges()
        )

    def choose_action_level_3(self, state, concept, episode):
        """Выбор действия на уровне 3 с использованием мета-правил."""
        if not self.meta_rules or concept is None:
            return self.choose_action_level_2(state, concept)

        # Ищем применимые правила
        applicable_rules = [rule for rule in self.meta_rules
                            if rule['condition'] == concept]

        if applicable_rules:
            # Выбираем правило с наивысшей уверенностью
            best_rule = max(applicable_rules, key=lambda x: x['confidence'])

            # Получаем лучшие действия для перехода в целевой концепт
            target_concept = best_rule['predicted_outcome']
            best_actions = self.get_best_actions_for_concept_transition(concept, target_concept)

            if best_actions:
                return np.random.choice(best_actions)

        # Fallback to level 2
        return self.choose_action_level_2(state, concept)

    def get_best_actions_for_concept_transition(self, concept_from, concept_to):
        """Находит действия, которые чаще всего ведут из concept_from в concept_to."""
        best_actions = []
        concept_states = self.concept_map.get(concept_from, [])

        action_scores = {}
        for state in concept_states:
            for action in range(self.num_actions):
                # Упрощенная эвристика: считаем "ценность" действия
                if (state, action) in self.transition_stats:
                    next_states = self.transition_stats[(state, action)]
                    for next_state, count in next_states.items():
                        next_concept = self.map_state_to_concept(next_state)
                        if next_concept == concept_to:
                            action_scores[action] = action_scores.get(action, 0) + count

        if action_scores:
            best_score = max(action_scores.values())
            best_actions = [action for action, score in action_scores.items()
                            if score >= best_score * 0.8]  # Все действия с близким к лучшему счетом

        return best_actions

    def update(self, state, action, reward, next_state, other_agents=None,
               env=None, episode=None):
        """Расширенный метод update с поддержкой уровня 3."""

        # Вызываем родительский update
        learning_signal = super().update(state, action, reward, next_state, other_agents)

        # Воспринимаем контекст среды
        if env is not None and episode is not None:
            self.perceive_environment_context(env, episode, reward)

        # Строим граф знаний и выводим правила на уровне 2+
        if self.level >= 2:
            self.build_knowledge_graph()
            self.infer_meta_rules()

        # Проверяем переход на уровень 3
        if (self.level == 2 and
                self.P > self.P_crit_level_3 and
                len(self.meta_rules) >= 3 and
                self.knowledge_graph.number_of_edges() >= 10):
            self.level = 3
            self.level_up_episode = episode
            print(f"Агент {self.id}: РОЖДЕНИЕ УРОВНЯ 3 на эпизоде {episode}!")