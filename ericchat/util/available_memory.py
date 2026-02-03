import psutil

def get_memory():
    vm = psutil.virtual_memory()
    available = vm.available
    available_gb = available/(1024*1024*1024)

    return available_gb
