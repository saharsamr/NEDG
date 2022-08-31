def is_array_of_empty_strings(arr):

    return True if len([e for e in arr if e != '']) == 0 else False


def sort_dict_by_key(dictionary, order='descending'):

    sorted_dict = sorted(dictionary.items(), key=lambda kv: (kv[1], kv[0]))
    return sorted_dict

