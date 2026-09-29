// Drives annotate.html with real mouse and key events and writes what happened into
// <pre id="selftest">. Loaded only with ?selftest=1; tests/test_annotate.py runs it in headless Chrome.
// The URL gives three walkable world points: ?a=x,y&b=x,y&c=x,y
"use strict";
(async () => {
  const q = new URLSearchParams(location.search);
  const pt = (k) => q.get(k).split(",").map(Number);
  const [A, B, C] = [pt("a"), pt("b"), pt("c")];
  const canvas = document.getElementById("map");
  const steps = [];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const until = async (test, what, ms = 8000) => {
    for (let t = 0; t < ms; t += 50) { if (test()) return; await sleep(50); }
    throw new Error("timed out waiting for " + what);
  };
  const settled = () => !app.saving && !app.dirty && app.routes.every((r) => r.plan);
  const mouse = (type, [x, y], extra = {}) => {
    const box = canvas.getBoundingClientRect();
    const target = type === "mouseup" ? window : canvas;
    target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX: box.left + x,
      clientY: box.top + y, button: extra.button || 0, shiftKey: !!extra.shift }));
  };
  const click = (p, extra) => { const s = toScreen(p); mouse("mousedown", s, extra); mouse("mouseup", s, extra); };
  const key = (k, extra = {}) => window.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, ...extra }));
  const check = (ok, what) => { steps.push((ok ? "ok   " : "FAIL ") + what); if (!ok) throw new Error(what); };
  const route = (id) => app.routes.find((r) => r.id === id);

  let result = { ok: false };
  try {
    await until(settled, "the scene's routes to be planned");
    const n0 = app.routes.length;
    // Zoom in on the three points, so a click lands within a few centimeters.
    app.routes.push({ id: "_view", start: A, via: [B], goal: C, plan: {} });
    centerOn(app.routes.pop()); draw();

    click(A); click(B);
    await until(settled, "the new route to be planned and saved");
    const id = app.selected, r = route(id);
    check(app.routes.length === n0 + 1 && r, "start + goal clicks make a route");
    check(r.plan.path && r.plan.path.length >= 2, "the route is planned: " + JSON.stringify(r.plan.facts || r.plan));
    check(Math.hypot(r.start[0] - A[0], r.start[1] - A[1]) < 0.2, "the start is where it was clicked");

    click(C, { shift: true });
    await until(settled, "the via point to be planned");
    check(r.via.length === 1, "shift+click adds a via point");

    const goal = toScreen(r.goal), to = toScreen(C);
    const along = (t) => [goal[0] + t * (to[0] - goal[0]), goal[1] + t * (to[1] - goal[1])];
    mouse("mousedown", goal); mouse("mousemove", along(0.15)); mouse("mousemove", along(0.3)); mouse("mouseup", along(0.3));
    await until(settled, "the moved goal to be planned");
    check(Math.hypot(r.goal[0] - B[0], r.goal[1] - B[1]) > 0.2, "dragging moves the goal");

    const via = toScreen(r.via[0]);
    canvas.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true,
      clientX: canvas.getBoundingClientRect().left + via[0], clientY: canvas.getBoundingClientRect().top + via[1] }));
    await until(settled, "the route without its via point to be planned");
    check(r.via.length === 0, "right-click removes a via point");

    const note = document.getElementById("note");
    note.value = "selftest"; note.dispatchEvent(new Event("input"));
    await sleep(600); await until(settled, "the note to be saved");
    check(r.note === "selftest", "the note is kept");

    click(C); click(B);  // not A: a click on a point selects that point
    await until(settled, "a second route");
    const second = app.selected;
    check(app.routes.length === n0 + 2 && second !== id, "a second route");
    key("Delete");
    await until(settled, "the deletion to be saved");
    check(!route(second) && app.routes.length === n0 + 1, "Delete removes the selected route");
    key("z", { ctrlKey: true });
    await until(settled, "the undo to be saved");
    check(!!route(second), "Ctrl+Z brings it back");
    app.selected = second; key("Delete");
    await until(settled, "the second deletion to be saved");
    check(!route(second), "deleted again");

    result = { ok: true, id, routes: app.routes.map(({ id, start, via, goal, note }) => ({ id, start, via, goal, note })),
               facts: r.plan.facts };
  } catch (e) {
    result = { ok: false, error: String(e.message || e) };
  }
  const pre = document.createElement("pre");
  pre.id = "selftest";
  pre.textContent = JSON.stringify({ ...result, steps });
  document.body.appendChild(pre);
})();
