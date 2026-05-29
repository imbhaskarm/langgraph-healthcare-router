from app.graph import build_graph


def main():
    print("Healthcare Support Assistant")
    print("Type your query below. Type 'quit' to exit.\n")

    agent = build_graph()

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        initial_state = {
            "customer_query":         query,
            "query_category":         "",
            "query_sentiment":        "",
            "escalation_customer_info": {},
            "on_call_support_info":   {},
            "final_response":         "",
        }

        result = agent.invoke(initial_state)

        print(f"\nCategory : {result['query_category']}")
        print(f"Sentiment: {result['query_sentiment']}")
        print(f"\nResponse:\n{result['final_response']}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
