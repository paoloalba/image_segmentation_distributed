
def parse_boolean(input_string):
    if input_string.lower().startswith("y"):
        return True
    elif input_string.lower().startswith("n"):
        return False
    else:
        raise Exception("Unrecognised boolean string: {0}".format(input_string))
