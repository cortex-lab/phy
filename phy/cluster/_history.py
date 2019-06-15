# -*- coding: utf-8 -*-

"""History class for undo stack."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# History class
#------------------------------------------------------------------------------

class History(object):
    """Implement a history of actions with an undo stack."""

    def __init__(self, base_item=None):
        self.clear(base_item)

    def clear(self, base_item=None):
        """Clear the history."""
        # List of changes, contains at least the base item.
        self._history = [base_item]
        # Index of the current item.
        self._index = 0

    @property
    def current_item(self):
        """Return the current element."""
        if self._history and self._index >= 0:
            self._check_index()
            return self._history[self._index]

    @property
    def current_position(self):
        """Current position in the history."""
        return self._index

    def _check_index(self):
        """Check that the index is without the bounds of _history."""
        assert 0 <= self._index <= len(self._history) - 1
        # There should always be the base item at least.
        assert len(self._history) >= 1

    def is_first(self):
        """Whether we are at the beginning of the stack."""
        return self._index == 1

    def is_last(self):
        """Whether we are at the end of the stack."""
        return self._index == len(self._history) - 1

    def iter(self, start=0, end=None):
        """Iterate through successive history items.

        Parameters
        ----------

        start : int
            Initial index for the loop.
        end : int
            Index of the last item to loop through + 1.

        """
        if end is None:
            end = self._index + 1
        elif end == 0:
            return
        if start >= end:
            return
        # Check arguments.
        assert 0 <= end <= len(self._history)
        assert 0 <= start <= end - 1
        for i in range(start, end):
            yield self._history[i]

    def __iter__(self):
        return self.iter()

    def __len__(self):
        return len(self._history)

    def add(self, item):
        """Add an item in the history."""
        self._check_index()
        # Possibly truncate the history up to the current point.
        self._history = self._history[:self._index + 1]
        # Append the item
        self._history.append(item)
        # Increment the index.
        self._index += 1
        self._check_index()
        # Check that the current element is what was provided to the function.
        assert id(self.current_item) == id(item)

    def back(self):
        """Go back in history if possible.

        Return the undone item.

        """
        if self._index <= 0:
            return None
        undone = self.current_item
        self._index -= 1
        self._check_index()
        return undone

    def undo(self):
        """Alias to back()."""
        return self.back()

    def forward(self):
        """Go forward in history if possible.

        Return the current item after going forward.

        """
        if self._index >= len(self._history) - 1:
            return None
        self._index += 1
        self._check_index()
        return self.current_item

    def redo(self):
        return self.forward()


class GlobalHistory(History):
    """Merge several controllers with different undo stacks."""

    def __init__(self, process_ups=None):
        super(GlobalHistory, self).__init__(())
        self.process_ups = process_ups

    def action(self, *controllers):
        """Register one or several controllers for this action."""
        self.add(tuple(controllers))

    def add_to_current_action(self, controller):
        """Add a controller to the current action."""
        item = self.current_item
        self._history[self._index] = item + (controller,)

    def undo(self):
        """Undo the last action.

        This will call `undo()` on all controllers involved in this action.

        """
        controllers = self.back()
        if controllers is None:
            ups = ()
        else:
            ups = tuple([controller.undo()
                        for controller in controllers])
        if self.process_ups is not None:
            return self.process_ups(ups)
        else:
            return ups

    def redo(self):
        """Redo the last action.

        This will call `redo()` on all controllers involved in this action.

        """
        controllers = self.forward()
        if controllers is None:
            ups = ()
        else:
            ups = tuple([controller.redo() for
                         controller in controllers])
        if self.process_ups is not None:
            return self.process_ups(ups)
        else:
            return ups
