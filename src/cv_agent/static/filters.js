/* Фильтры столбцов таблицы вакансий в стиле Excel:
   клик по заголовку открывает панель со списком значений и чекбоксами. */
(function () {
  "use strict";

  var configEl = document.getElementById("filter-config");
  if (!configEl) return;
  var config;
  try {
    config = JSON.parse(configEl.textContent);
  } catch (e) {
    return;
  }
  var options = config.options || {};
  var active = config.active || {};
  var separator = config.separator || "||";
  var paramPrefix = config.paramPrefix || "f_";

  var openPanel = null;

  function closePanel() {
    if (openPanel) {
      openPanel.remove();
      openPanel = null;
    }
  }

  function currentParams() {
    return new URLSearchParams(window.location.search);
  }

  function applyFilter(key, selected) {
    var params = currentParams();
    params.delete("page");
    if (selected.length) {
      params.set(paramPrefix + key, selected.join(separator));
    } else {
      params.delete(paramPrefix + key);
    }
    var qs = params.toString();
    window.location.href = window.location.pathname + (qs ? "?" + qs : "");
  }

  function buildCompRow(key) {
    if (key !== "score") return null;
    var row = document.createElement("div");
    row.className = "filter-comp-row";
    var ops = ["<", "=", ">"];
    var opButtons = {};
    ops.forEach(function (op) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "filter-mini filter-op";
      b.textContent = op;
      b.addEventListener("click", function () {
        var wasActive = b.classList.contains("active");
        ops.forEach(function (o) { opButtons[o].classList.remove("active"); });
        if (!wasActive) b.classList.add("active");
      });
      opButtons[op] = b;
      row.appendChild(b);
    });
    var numInput = document.createElement("input");
    numInput.type = "number";
    numInput.className = "filter-number";
    numInput.placeholder = "Оценка";
    numInput.min = "0";
    numInput.max = "100";
    row.appendChild(numInput);

    (active[key] || []).forEach(function (value) {
      var m = value.match(/^\s*([<>=])\s*(\d{1,3}(?:\.\d)?)\s*$/);
      if (m && opButtons[m[1]]) {
        opButtons[m[1]].classList.add("active");
        numInput.value = m[2];
      }
    });

    row._compValue = function () {
      var chosenOp = null;
      ops.forEach(function (o) {
        if (opButtons[o].classList.contains("active")) chosenOp = o;
      });
      var raw = numInput.value.trim();
      if (!chosenOp || raw === "") return null;
      return chosenOp + raw;
    };
    return row;
  }

  function buildPanel(key, th) {
    var panel = document.createElement("div");
    panel.className = "filter-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Фильтр: " + key);

    var search = document.createElement("input");
    search.type = "text";
    search.className = "filter-search";
    search.placeholder = "Поиск...";

    var compRow = buildCompRow(key);

    var actions = document.createElement("div");
    actions.className = "filter-actions";
    var selectAll = document.createElement("button");
    selectAll.type = "button";
    selectAll.className = "filter-mini";
    selectAll.textContent = "Выделить всё";
    var clearSel = document.createElement("button");
    clearSel.type = "button";
    clearSel.className = "filter-mini";
    clearSel.textContent = "Снять выделение";
    actions.appendChild(selectAll);
    actions.appendChild(clearSel);

    var list = document.createElement("div");
    list.className = "filter-list";

    var values = options[key] || [];
    var selectedSet = new Set(active[key] || []);
    values.forEach(function (value) {
      var label = document.createElement("label");
      label.className = "filter-option";
      var box = document.createElement("input");
      box.type = "checkbox";
      box.value = value;
      box.checked = selectedSet.has(value);
      label.appendChild(box);
      label.appendChild(document.createTextNode(" " + value));
      list.appendChild(label);
    });

    var buttons = document.createElement("div");
    buttons.className = "filter-buttons";
    var ok = document.createElement("button");
    ok.type = "button";
    ok.className = "btn small filter-apply";
    ok.textContent = "OK";
    var cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "btn small filter-cancel";
    cancel.textContent = "Отмена";
    buttons.appendChild(ok);
    buttons.appendChild(cancel);

    panel.appendChild(search);
    if (compRow) panel.appendChild(compRow);
    panel.appendChild(actions);
    panel.appendChild(list);
    panel.appendChild(buttons);

    search.addEventListener("input", function () {
      var q = search.value.trim().toLowerCase();
      Array.prototype.forEach.call(list.children, function (label) {
        var text = label.textContent.toLowerCase();
        label.style.display = !q || text.indexOf(q) !== -1 ? "" : "none";
      });
    });
    selectAll.addEventListener("click", function () {
      Array.prototype.forEach.call(list.querySelectorAll("input"), function (box) {
        if (box.closest("label").style.display !== "none") box.checked = true;
      });
    });
    clearSel.addEventListener("click", function () {
      Array.prototype.forEach.call(list.querySelectorAll("input"), function (box) {
        box.checked = false;
      });
    });
    ok.addEventListener("click", function () {
      var chosen = Array.prototype.map.call(
        list.querySelectorAll("input:checked"),
        function (box) { return box.value; }
      );
      if (compRow) {
        var comp = compRow._compValue();
        if (comp) chosen.push(comp);
      }
      applyFilter(key, chosen);
    });
    cancel.addEventListener("click", closePanel);

    return panel;
  }

  function openFor(th) {
    var key = th.getAttribute("data-filter");
    if (!key || !(key in options)) return;
    if (openPanel && openPanel._forKey === key) {
      closePanel();
      return;
    }
    closePanel();
    var panel = buildPanel(key, th);
    panel._forKey = key;
    document.body.appendChild(panel);
    var rect = th.getBoundingClientRect();
    var width = Math.max(rect.width, 240);
    panel.style.minWidth = width + "px";
    var left = Math.min(rect.left, window.innerWidth - width - 8);
    panel.style.left = Math.max(8, left) + "px";
    var top = rect.bottom + 4;
    if (top + panel.offsetHeight > window.innerHeight - 8) {
      top = Math.max(8, window.innerHeight - panel.offsetHeight - 8);
    }
    panel.style.top = top + "px";
    openPanel = panel;
    var search = panel.querySelector(".filter-search");
    if (search) search.focus();
  }

  Array.prototype.forEach.call(
    document.querySelectorAll("th.filterable"),
    function (th) {
      th.addEventListener("click", function () { openFor(th); });
    }
  );

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closePanel();
  });
  document.addEventListener("click", function (event) {
    if (!openPanel) return;
    if (openPanel.contains(event.target)) return;
    if (event.target.closest && event.target.closest("th.filterable")) return;
    closePanel();
  });
})();
