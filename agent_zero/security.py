def anonymize_data(data: str) -> str:
    """
    Simple anonymization function. In a real-world scenario, 
    this would be much more sophisticated.
    """
    # This is a very basic implementation. In practice, you'd use more 
    # advanced techniques to identify and anonymize sensitive information.
    sensitive_terms = ['name', 'email', 'phone', 'address', 'ssn']
    for term in sensitive_terms:
        if term in data.lower():
            data = data.replace(data.split()[data.lower().split().index(term) + 1], '[REDACTED]')
    return data