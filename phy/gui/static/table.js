
var Table = function (el) {
    this.el = el;
    this.state = {
        sortCol: null,
        sortDir: null,
        selected: [],
        pinned: [],
    }
    this.headers = {};  // {name: th} mapping
    this.rows = {};  // {id: tr} mapping
    this.tablesort = null;
};

Table.prototype.setData = function(data) {
    if (data.items.length == 0) return;
    var that = this;
    var keys = data.cols;

    var thead = document.createElement("thead");
    var tbody = document.createElement("tbody");

    // Header.
    var tr = document.createElement("tr");
    for (var j = 0; j < keys.length; j++) {
        var key = keys[j];
        var th = document.createElement("th");
        th.appendChild(document.createTextNode(key));
        tr.appendChild(th);
        this.headers[key] = th;
    }
    thead.appendChild(tr);

    // Data rows.
    for (var i = 0; i < data.items.length; i++) {
        tr = document.createElement("tr");
        var row = data.items[i];
        for (var j = 0; j < keys.length; j++) {
            var key = keys[j];
            var value = row[key];
            var td = document.createElement("td");
            td.appendChild(document.createTextNode(value));
            tr.appendChild(td);
        }

        // Set the data values on the row.
        for (var key in row) {
            tr.dataset[key] = row[key];
        }

        tr.onclick = function(e) {
            var selected = [parseInt(this.dataset.id)];

            var evt = e ? e:window.event;
            if (evt.ctrlKey || evt.metaKey) {
                selected = that.state.selected.concat(selected);
            }
            that.select(selected);
        }

        tbody.appendChild(tr);
        this.rows[data.items[i].id] = tr;
    }

    this.el.appendChild(thead);
    this.el.appendChild(tbody);

    // Enable the tablesort plugin.
    this.tablesort = new Tablesort(this.el);

    // Synchronize the state.
    var that = this;
    this.el.addEventListener('afterSort', function() {
        for (var header in that.headers) {
            if (that.headers[header].classList.contains('sort-up')) {
                that.state.sortCol = header;
                that.state.sortDir = 'desc';
                break;
            }
            if (that.headers[header].classList.contains('sort-down')) {
                that.state.sortCol = header;
                that.state.sortDir = 'asc';
                break;
            }
        }
    });
};

Table.prototype.setState = function(state) {

    // Make sure both sortCol and sortDir are specified.
    if (!('sortCol' in state) && ('sortDir' in state)) {
        state.sortCol = this.state.sortCol;
    }
    if (!('sortDir' in state) && ('sortCol' in state)) {
        state.sortDir = this.state.sortDir;
    }

    if ('sortCol' in state) {

        // Update the state.
        this.state.sortCol = state.sortCol;
        this.state.sortDir = state.sortDir;

        // Remove all sorts.
        for (var h in this.headers) {
            this.headers[h].classList.remove('sort-up');
            this.headers[h].classList.remove('sort-down');
        }

        // Set the sort direction in the header class.
        var header = this.headers[state.sortCol];
        header.classList.add(state.sortDir === 'desc' ?
                             'sort-down' : 'sort-up');

        // Sort the table.
        this.tablesort.sortTable(header);
    }
    if ('selected' in state) {
        this.setRowClass('selected', state.selected);
        this.state.selected = state.selected.map(parseInt);
    }
    if ('pinned' in state) {
        this.setRowClass('pinned', state.pinned);
        this.state.pinned = state.pinned.map(parseInt);
    }
};

Table.prototype.setRowClass = function(name, ids) {
    // Remove the class on all rows.
    for (var i = 0; i < this.state[name].length; i++) {
        var id = this.state[name][i];
        var row = this.rows[id];
        row.classList.remove(name);
    }

    // Add the class.
    for (var i = 0; i < ids.length; i++) {
        var id = ids[i];
        this.rows[id].classList.add(name);
    }
};

Table.prototype.stateUpdated = function() {
    // TODO: call widget.setState(this.state);
};

Table.prototype.getState = function() {
    return this.state;
}

Table.prototype.clear = function() {
    this.setState({
        selected: [],
        pinned: [],
    });
};

Table.prototype.select = function(items) {
    this.setState({
        selected: items,
    });
};

Table.prototype.pin = function(items) {
    this.setState({
        pinned: items,
    });
};

Table.prototype.unpin = function() {
    this.setState({
        pinned: [],
    });
};

Table.prototype.next = function() {
    if (this.state.selected.length != 1) return;
    var id = this.state.selected[0];
    var row = this.rows[id];
    var i0 = row.rowIndex + 1;
    var items = [];

    for (var i = i0; i < this.el.rows.length; i++) {
        row = this.el.rows[i];
        if (!(row.dataset.skip)) {
            items.push(row.dataset.id);
            break;
        }
    }

    if (!(items.length)) return;

    // TODO: keep the pinned
    this.setState({
        selected: items,
    });
};

Table.prototype.previous = function() {
    if (this.state.selected.length != 1) return;
    var id = this.state.selected[0];
    var row = this.rows[id];
    var i0 = row.rowIndex - 1;
    var items = [];

    // NOTE: i >= 1 because we skip the header column.
    for (var i = i0; i >= 1; i--) {
        row = this.el.rows[i];
        if (!(row.dataset.skip)) {
            items.push(row.dataset.id);
            break;
        }
    }

    if (!(items.length)) return;

    // TODO: keep the pinned
    this.setState({
        selected: items,
    });
};
