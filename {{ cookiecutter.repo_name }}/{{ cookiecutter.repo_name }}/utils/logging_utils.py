from colorama import init, Fore, Back, Style


def print_bright(s):

    init()
    print(Style.BRIGHT + s + Style.RESET_ALL)


def print_green(info, value=""):

    print(Fore.GREEN + "[%s] " % info + Style.RESET_ALL + str(value))


def print_red(info, value=""):

    print(Fore.RED + "[%s] " % info + Style.RESET_ALL + str(value))


def str_to_bluestr(string):

    return Fore.BLUE + "%s" % string + Style.RESET_ALL


def str_to_yellowstr(string):

    return Fore.YELLOW + "%s" % string + Style.RESET_ALL