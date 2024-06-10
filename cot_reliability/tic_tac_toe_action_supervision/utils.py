def printc(text, color):
    """
    Prints the given text in the specified color.

    :param text: The text to be printed
    :param color: The color in which the text is to be printed. 
                  Accepts 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }

    # Check if the specified color is valid
    if color not in colors:
        print("Invalid color. Choose from 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.")
        return

    # Print the text in the specified color
    print(f"{colors[color]}{text}\033[0m")
