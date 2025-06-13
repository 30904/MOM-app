import React, { useState, useCallback, memo } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { FaPlus } from 'react-icons/fa';

const TaskItem = memo(({ task, index }) => (
  <Draggable key={task.id} draggableId={task.id} index={index}>
    {(provided, snapshot) => (
      <div
        ref={provided.innerRef}
        {...provided.draggableProps}
        {...provided.dragHandleProps}
        className={`p-2 rounded-lg transition-all duration-200 ${
          snapshot.isDragging
            ? 'bg-white shadow-lg scale-105'
            : 'bg-white hover:shadow-md'
        }`}
        style={{
          ...provided.draggableProps.style,
          transform: snapshot.isDragging
            ? provided.draggableProps.style?.transform
            : 'none',
        }}
      >
        <div className="flex flex-col space-y-1">
          <p className="text-sm text-gray-700">{task.content}</p>
          <div className="flex justify-between items-center">
            <span className={`text-xs px-2 py-0.5 rounded-full ${getPriorityColor(task.priority)}`}>
              {task.priority}
            </span>
            <button className="text-gray-400 hover:text-gray-600">
              <FaPlus className="h-3 w-3" />
            </button>
          </div>
        </div>
      </div>
    )}
  </Draggable>
));

const Column = memo(({ columnId, tasks, title }) => (
  <Droppable droppableId={columnId}>
    {(provided, snapshot) => (
      <div
        className={`flex flex-col h-full rounded-lg transition-colors duration-200 ${
          snapshot.isDraggingOver ? 'bg-blue-50' : 'bg-gray-50'
        }`}
        {...provided.droppableProps}
        ref={provided.innerRef}
      >
        <div className="p-2 border-b border-gray-200 bg-white rounded-t-lg">
          <div className="flex justify-between items-center">
            <h4 className="text-sm font-medium text-gray-700">{title}</h4>
            <span className="text-xs text-gray-500">{tasks.length} items</span>
          </div>
        </div>
        <div className="flex-1 p-1.5 space-y-1.5 overflow-y-auto max-h-[calc(100vh-25rem)]">
          {tasks.map((task, index) => (
            <TaskItem key={task.id} task={task} index={index} />
          ))}
          {provided.placeholder}
        </div>
      </div>
    )}
  </Droppable>
));

const getPriorityColor = (priority) => {
  switch (priority) {
    case 'high':
      return 'bg-red-100 text-red-700';
    case 'medium':
      return 'bg-yellow-100 text-yellow-700';
    case 'low':
      return 'bg-green-100 text-green-700';
    default:
      return 'bg-gray-100 text-gray-700';
  }
};

const getColumnTitle = (columnId) => {
  switch (columnId) {
    case 'todo':
      return 'To Do';
    case 'inProgress':
      return 'In Progress';
    case 'done':
      return 'Done';
    default:
      return columnId;
  }
};

const ActionItemsSection = () => {
  const [tasks, setTasks] = useState({
    todo: [
      { id: 'task-1', content: 'Prepare presentation slides', priority: 'high' },
      { id: 'task-2', content: 'Schedule follow-up meeting', priority: 'medium' },
    ],
    inProgress: [
      { id: 'task-3', content: 'Draft project timeline', priority: 'high' },
    ],
    done: [
      { id: 'task-4', content: 'Send meeting notes', priority: 'low' },
    ],
  });

  const onDragEnd = useCallback((result) => {
    const { source, destination } = result;
    if (!destination) return;
    if (source.droppableId === destination.droppableId && source.index === destination.index) return;

    setTasks(prevTasks => {
      const sourceTasks = [...prevTasks[source.droppableId]];
      const [movedTask] = sourceTasks.splice(source.index, 1);
      const destinationTasks = [...prevTasks[destination.droppableId]];
      destinationTasks.splice(destination.index, 0, movedTask);

      return {
        ...prevTasks,
        [source.droppableId]: sourceTasks,
        [destination.droppableId]: destinationTasks,
      };
    });
  }, []);

  return (
    <div className="h-full flex flex-col">
      <div className="p-2 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">Action Items</h3>
      </div>
      <div className="flex-1 p-2 overflow-hidden">
        <DragDropContext onDragEnd={onDragEnd}>
          <div className="h-full grid grid-cols-1 lg:grid-cols-3 gap-2 min-h-0">
            {['todo', 'inProgress', 'done'].map((columnId) => (
              <Column
                key={columnId}
                columnId={columnId}
                tasks={tasks[columnId]}
                title={getColumnTitle(columnId)}
              />
            ))}
          </div>
        </DragDropContext>
      </div>
    </div>
  );
};

export default memo(ActionItemsSection);