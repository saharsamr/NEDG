def is_array_of_empty_strings(arr):

    return True if len([e for e in arr if e != '']) == 0 else False
