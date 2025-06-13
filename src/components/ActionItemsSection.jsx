import React, { useState } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';

const initialTasks = {
  'To Do': [
    { id: '1', content: 'Prepare slides', assignee: 'Piyush', dueDate: '2025-06-10', priority: 'high' },
    { id: '2', content: 'Schedule meeting', assignee: 'Arnav', dueDate: '2025-06-15', priority: 'medium' },
  ],
  'In Progress': [],
  'Completed': [],
};

const ActionItemsSection = () => {
  const [tasks, setTasks] = useState(initialTasks);

  const onDragEnd = (result) => {
    const { source, destination } = result;
    if (!destination) return;

    const sourceColumn = source.droppableId;
    const destColumn = destination.droppableId;
    const sourceItems = [...tasks[sourceColumn]];
    const destItems = [...tasks[destColumn]];
    const [movedItem] = sourceItems.splice(source.index, 1);
    destItems.splice(destination.index, 0, movedItem);

    setTasks({
      ...tasks,
      [sourceColumn]: sourceItems,
      [destColumn]: destItems,
    });
  };

  const getDueDateColor = (dueDate) => {
    const today = new Date('2025-06-12');
    const due = new Date(dueDate);
    if (due < today) return 'text-red-500'; // Overdue
    if (due.toDateString() === today.toDateString()) return 'text-yellow-500'; // Upcoming
    return 'text-green-500'; // Completed
  };

  const getInitials = (name) => {
    return name.charAt(0).toUpperCase();
  };

  return (
    <DragDropContext onDragEnd={onDragEnd}>
      <div className="p-4 bg-white dark:bg-gray-800 shadow-md rounded-lg animate-componentFadeIn">
        <div className="flex space-x-4">
          {Object.keys(tasks).map((column) => (
            <Droppable droppableId={column} key={column}>
              {(provided) => (
                <div
                  className="flex-1 p-3 bg-gray-100 dark:bg-gray-700 rounded-lg"
                  {...provided.droppableProps}
                  ref={provided.innerRef}
                >
                  <h3 className="text-lg font-semibold mb-2">{column}</h3>
                  {tasks[column].map((task, index) => (
                    <Draggable key={task.id} draggableId={task.id} index={index}>
                      {(provided) => (
                        <div
                          ref={provided.innerRef}
                          {...provided.draggableProps}
                          {...provided.dragHandleProps}
                          className="p-3 mb-2 bg-white dark:bg-gray-600 shadow rounded-lg transition-transform duration-200"
                        >
                          <p>{task.content}</p>
                          <div className="flex items-center space-x-2 mt-1">
                            <div className="h-6 w-6 bg-blue-900 text-white rounded-full flex items-center justify-center text-xs">
                              {getInitials(task.assignee)}
                            </div>
                            <span className="text-sm text-gray-500">{task.assignee}</span>
                            <span className={`text-sm ${getDueDateColor(task.dueDate)}`}>{task.dueDate}</span>
                            <span className={`h-4 w-4 rounded-full ${task.priority === 'high' ? 'bg-red-500' : 'bg-yellow-500'}`}></span>
                          </div>
                        </div>
                      )}
                    </Draggable>
                  ))}
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          ))}
        </div>
      </div>
    </DragDropContext>
  );
};

export default ActionItemsSection;