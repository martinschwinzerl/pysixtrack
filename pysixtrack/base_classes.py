from dataclasses import dataclass


class _MetaElement(type):
    def __new__(cls, clsname, bases, dct):
        description = dct.get('_description', [])
        extra = dct.get('_extra', [])
        dct['_base_fields'] = [dd[0] for dd in description]
        dct['_extra_fields'] = [dd[0] for dd in extra]
        dct['_fields'] = dct['_base_fields'] + dct['_extra_fields']
        ann = {}
        dct['__annotations__'] = ann
        for dd in description:
            ann[dd[0]] = type(dd[3])
            dct[dd[0]] = dd[3]
        for dd in extra:
            ann[dd[0]] = type(dd[3])
            dct[dd[0]] = dd[3]
        try:
            doc = [dct['__doc__'], '\nFields:\n']
        except KeyError:
            doc = ['\nFields:\n']
        fields = [f"{field:10} [{unit+']:':5} {desc} " for field,
                  unit, desc, default in description]
        fields += [f"{field:10} [{unit+']:':5} {desc} " for field,
                   unit, desc, default in extra]
        doc += fields
        dct['__doc__'] = "\n".join(doc)
        newclass = super(_MetaElement, cls).__new__(cls, clsname, bases, dct)
        return dataclass(newclass)


class Base(metaclass=_MetaElement):

    def get_fields(self, keepextra=False):
        if keepextra:
            return self.__class__._fields
        else:
            return self.__class__._base_fields

    def to_dict(self, keepextra=False):
        out={kk: getattr(self, kk) for kk in self.get_fields(keepextra)}
        out['__class__']=self.__class__.__name__
        return out

    @classmethod
    def from_dict(cls, dct, keepextra=True):
        self=cls()
        for kk in cls._base_fields:
            setattr(self, kk, dct[kk])
        if keepextra:
            for kk in cls._extra_fields:
                if kk in dct:
                    setattr(self, kk, dct[kk])
        return self

    def copy(self, keepextra=True):
        return self.__class__.from_dict(self.to_dict(keepextra), keepextra)


class Element(Base):
    pass
