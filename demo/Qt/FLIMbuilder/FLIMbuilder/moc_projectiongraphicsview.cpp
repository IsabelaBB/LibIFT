/****************************************************************************
** Meta object code from reading C++ file 'projectiongraphicsview.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "projectiongraphicsview.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'projectiongraphicsview.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ProjectionGraphicsView_t {
    QByteArrayData data[9];
    char stringdata0[180];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ProjectionGraphicsView_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ProjectionGraphicsView_t qt_meta_stringdata_ProjectionGraphicsView = {
    {
QT_MOC_LITERAL(0, 0, 22), // "ProjectionGraphicsView"
QT_MOC_LITERAL(1, 23, 30), // "mouseContextMenuRequest_signal"
QT_MOC_LITERAL(2, 54, 0), // ""
QT_MOC_LITERAL(3, 55, 18), // "QContextMenuEvent*"
QT_MOC_LITERAL(4, 74, 5), // "event"
QT_MOC_LITERAL(5, 80, 27), // "sceneAreaSizeChanged_signal"
QT_MOC_LITERAL(6, 108, 32), // "updateSceneAutoReprojection_slot"
QT_MOC_LITERAL(7, 141, 13), // "autoReproject"
QT_MOC_LITERAL(8, 155, 24) // "updateNodesPosition_slot"

    },
    "ProjectionGraphicsView\0"
    "mouseContextMenuRequest_signal\0\0"
    "QContextMenuEvent*\0event\0"
    "sceneAreaSizeChanged_signal\0"
    "updateSceneAutoReprojection_slot\0"
    "autoReproject\0updateNodesPosition_slot"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ProjectionGraphicsView[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   34,    2, 0x06 /* Public */,
       5,    0,   37,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       6,    1,   38,    2, 0x0a /* Public */,
       8,    0,   41,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, QMetaType::Bool,    7,
    QMetaType::Void,

       0        // eod
};

void ProjectionGraphicsView::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ProjectionGraphicsView *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->mouseContextMenuRequest_signal((*reinterpret_cast< QContextMenuEvent*(*)>(_a[1]))); break;
        case 1: _t->sceneAreaSizeChanged_signal(); break;
        case 2: _t->updateSceneAutoReprojection_slot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 3: _t->updateNodesPosition_slot(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ProjectionGraphicsView::*)(QContextMenuEvent * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProjectionGraphicsView::mouseContextMenuRequest_signal)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (ProjectionGraphicsView::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProjectionGraphicsView::sceneAreaSizeChanged_signal)) {
                *result = 1;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject ProjectionGraphicsView::staticMetaObject = { {
    QMetaObject::SuperData::link<QGraphicsView::staticMetaObject>(),
    qt_meta_stringdata_ProjectionGraphicsView.data,
    qt_meta_data_ProjectionGraphicsView,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ProjectionGraphicsView::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ProjectionGraphicsView::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ProjectionGraphicsView.stringdata0))
        return static_cast<void*>(this);
    return QGraphicsView::qt_metacast(_clname);
}

int ProjectionGraphicsView::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGraphicsView::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void ProjectionGraphicsView::mouseContextMenuRequest_signal(QContextMenuEvent * _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void ProjectionGraphicsView::sceneAreaSizeChanged_signal()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
