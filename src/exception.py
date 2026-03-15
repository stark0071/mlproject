import sys

def error_message_detail(error, error_detail: sys):
    """
    This function extracts detailed information about an exception.
    - 'error': The actual exception object captured in the 'except' block.
    - 'error_detail': The 'sys' module, which provides access to the traceback.
    """
    
    # sys.exc_info() returns a tuple: (type, value, traceback)
    # We use underscores (_) for the first two because we only need the traceback object (exc_tb).
    _, _, exc_tb = error_detail.exc_info()

    # We access the 'tb_frame' (the execution state) to get the 'f_code' (the compiled code)
    # and finally 'co_filename' to find the exact file where the error happened.
    file_name = exc_tb.tb_frame.f_code.co_filename

    # exc_tb.tb_lineno provides the specific line number where the code crashed.
    line_number = exc_tb.tb_lineno

    # We build a final string that combines the file name, line number, and the error message.
    # We convert 'error' to a string to ensure it can be joined with the text.
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
