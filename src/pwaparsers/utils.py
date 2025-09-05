def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

def flat_tuple(tpl):
    # flatten tuple
    if isinstance(tpl, tuple) or isinstance(tpl, list):
        for item in tpl:
            yield from flat_tuple(item)
    else:
        yield tpl

def flat_sorted_tuple(tpl):
    return tuple(sorted(
        flat_tuple(tpl)
        )
    )
