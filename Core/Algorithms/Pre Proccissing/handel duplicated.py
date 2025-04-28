# Function to handle duplicates in specific columns or all
def handle_duplicates(data, columns=None, keep='first'):
    if columns:
        data[columns] = data[columns].drop_duplicates(keep=keep)
    else:
        data = data.drop_duplicates(keep=keep)
    return data