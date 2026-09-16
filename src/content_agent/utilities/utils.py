def split_user_full_name(user_full_name: str) -> tuple[str, str]:
    """Split a full name into first and last name.

    Example:
        "Faeze Saghafi" -> ("Faeze", "Saghafi")
        "Cher" -> ("Cher", "")
    """

    name_parts = user_full_name.strip().split(" ", 1)
    first_name = name_parts[0]
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    return first_name, last_name