from IPython.display import Markdown

def printmd(text):
    """
    Prints Markdown-formatted text in JupyterLab notebooks.
    
    Args:
        text (str): The Markdown-formatted text to be printed.
    """
    display(Markdown(text))