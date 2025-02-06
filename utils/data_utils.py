def parse_manifold_config(manifold_config_str):
    """
    Parses the manifold configuration string and returns a list of tuples (manifold_type, dimension).
    Example input: "H8H12S4S4S4E4"
    """
    import re
    pattern = r'([HSE])(\d+)'
    matches = re.findall(pattern, manifold_config_str)
    manifolds_config = []
    for match in matches:
        manifold_type = {'H': 'hyperbolic', 'S': 'spherical', 'E': 'euclidean'}[match[0]]
        dim = int(match[1])
        manifolds_config.append((manifold_type, dim))
    return manifolds_config