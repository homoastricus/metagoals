from simulation.simulation import simulate_advanced
from simulation.simulation import simulate_advanced

if __name__ == "__main__":
    print("Запуск эволюционной симуляции с кооперацией...")

    # Теперь функция возвращает словарь с результатами
    results = simulate_advanced(episodes=750, initial_num_agents=15)

    # Извлекаем результаты из словаря
    agents = results['agents']
    critical_episodes = results['critical_episodes']
    activation_stats = results['activation_stats']
    reproduction_events = results['reproduction_events']
    death_events = results['death_events']
    cooperation_events = results['cooperation_events']

    print(f"\nСимуляция завершена!")
    print(f"Всего агентов: {len(agents)}")
    print(f"Из них живых: {len([a for a in agents if a.alive])}")
    print(f"Событий размножения: {len(reproduction_events)}")
    print(f"Событий смерти: {len(death_events)}")
    print(f"Кооперативных взаимодействий: {len(cooperation_events)}")