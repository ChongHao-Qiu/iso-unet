import torch
import colorama as ca

def check_cuda():
    p_hint(f'CUDA availability: {torch.cuda.is_available()}')
    p_hint(f'Available GPUs: {torch.cuda.device_count()}')

def p_header(text):
    print(ca.Fore.CYAN + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_hint(text):
    print(ca.Fore.LIGHTBLACK_EX + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_success(text):
    print(ca.Fore.GREEN + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_fail(text):
    print(ca.Fore.RED + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_warning(text):
    print(ca.Fore.YELLOW + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def wrap_lon(t):
    L = t.shape[-1]
    half = L // 2
    left = t[..., :half]
    right = t[..., half:]
    return torch.cat([right, t, left], dim=-1)

def unwrap_lon(t):
    """
    Reverse the wrap_lon() transformation:
    wrapped = [right-half | original | left-half]
    """
    L2 = t.shape[-1]
    L = L2 // 2
    half = L // 2
    return t[..., half:half+L]