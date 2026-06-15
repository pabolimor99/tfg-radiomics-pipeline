import os
import csv
import slicer
import qt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "pairs", "longitudinal_pairs.csv")

OUTCOME_CSV = os.path.join(PROJECT_ROOT, "outputs", "review", "classification_outcomes.csv")

REQUIRE_RADIOMICS_VALID = True
cases = []
current_case_index = -1
current_case_uid = None
reviewer_widget = None

def _parse_bool(value):
    if value is None:
        return False
    return str(value).strip().lower() in ("true", "1", "yes", "y", "mejora")


def _resolve_path(path_str):
    if path_str is None:
        return None
    path_str = str(path_str).strip()
    if path_str == "":
        return None
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_str))


def _read_csv_rows(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _normalize_timepoint(value):
    value = str(value or "").strip().lower()
    if value in ("baseline", "base", "bl"):
        return "baseline"
    if value in ("followup", "follow-up", "fu"):
        return "followup"
    return None


def _build_cases_from_long_rows(rows):
    if rows:
        required_columns = {
            "pair_uid",
            "patient_id",
            "pair_index",
            "timepoint",
            "image_path",
            "mask_path",
            "is_radiomics_valid",
        }
        missing_columns = required_columns - set(rows[0].keys())
        if missing_columns:
            raise ValueError(f"Faltan columnas en {CSV_PATH}: {sorted(missing_columns)}")

    pairs = {}

    for row in rows:
        timepoint = _normalize_timepoint(row.get("timepoint"))
        if timepoint not in ("baseline", "followup"):
            continue

        patient_id = str(row.get("patient_id", "")).strip()
        pair_index = str(row.get("pair_index", "")).strip()

        if patient_id == "":
            continue

        if pair_index == "":
            pair_index = "0"

        pair_uid = str(row.get("pair_uid", "")).strip()
        if pair_uid == "":
            raise ValueError(f"Fila sin pair_uid para patient_id={patient_id}, pair_index={pair_index}")

        pair = pairs.setdefault(
            pair_uid,
            {
                "uid": pair_uid,
                "patient_id": patient_id,
                "pair_index": pair_index,
                "baseline": None,
                "followup": None,
                "is_radiomics_valid": True,
                "rows": {},
            },
        )

        if "is_radiomics_valid" in row and not _parse_bool(row.get("is_radiomics_valid")):
            pair["is_radiomics_valid"] = False

        pair["rows"][timepoint] = row
        pair[timepoint] = {
            "image": _resolve_path(row.get("image_path")),
            "mask": _resolve_path(row.get("mask_path")),
        }

    parsed_cases = []

    for pair in pairs.values():
        if REQUIRE_RADIOMICS_VALID and not pair["is_radiomics_valid"]:
            continue

        baseline = pair["baseline"]
        followup = pair["followup"]
        has_baseline = baseline and baseline["image"] and baseline["mask"]
        has_followup = followup and followup["image"] and followup["mask"]

        if not has_baseline or not has_followup:
            continue

        parsed_cases.append({
            "uid": pair["uid"],
            "patient_id": pair["patient_id"],
            "pair_index": pair["pair_index"],
            "baseline_image": baseline["image"],
            "baseline_mask": baseline["mask"],
            "followup_image": followup["image"],
            "followup_mask": followup["mask"],
            "row": pair["rows"],
        })

    return parsed_cases


def _load_volume(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: no existe")
    node = slicer.util.loadVolume(path)
    if node is None:
        raise RuntimeError(f"{path}: no se pudo cargar")
    return node


def _load_segmentation(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: no existe")
    node = slicer.util.loadSegmentation(path)
    if node is None:
        raise RuntimeError(f"{path}: no se pudo cargar")
    return node


def _set_two_slice_layout():
    layout_manager = slicer.app.layoutManager()
    layout_node = slicer.app.layoutManager().layoutLogic().GetLayoutNode()

    custom_layout = """
    <layout type="horizontal">
      <item splitSize="500">
        <view class="vtkMRMLSliceNode" singletontag="Red">
          <property name="orientation" action="default">Axial</property>
          <property name="viewlabel" action="default">BL</property>
          <property name="viewcolor" action="default">#F34A33</property>
        </view>
      </item>
      <item splitSize="500">
        <view class="vtkMRMLSliceNode" singletontag="Yellow">
          <property name="orientation" action="default">Axial</property>
          <property name="viewlabel" action="default">FU</property>
          <property name="viewcolor" action="default">#EDD54C</property>
        </view>
      </item>
    </layout>
    """

    custom_layout_id = 501
    layout_node.AddLayoutDescription(custom_layout_id, custom_layout)
    layout_manager.setLayout(custom_layout_id)


def _configure_slice_view(slice_name, volume_node, orientation="Axial"):
    slice_widget = slicer.app.layoutManager().sliceWidget(slice_name)
    slice_logic = slice_widget.sliceLogic()
    slice_node = slice_logic.GetSliceNode()
    composite_node = slice_logic.GetSliceCompositeNode()

    slice_node.SetOrientation(orientation)
    composite_node.SetBackgroundVolumeID(volume_node.GetID())


def _show_segmentation_only_in_view(segmentation_node, slice_name, fill_opacity=0.5, outline_opacity=1.0):
    display_node = segmentation_node.GetDisplayNode()

    display_node.RemoveAllViewNodeIDs()
    slice_node = slicer.util.getNode(f"vtkMRMLSliceNode{slice_name}")
    display_node.AddViewNodeID(slice_node.GetID())

    display_node.SetVisibility2D(True)
    display_node.SetVisibility3D(False)
    display_node.SetOpacity2DFill(fill_opacity)
    display_node.SetOpacity2DOutline(outline_opacity)
    display_node.SetAllSegmentsVisibility(True)


def _fit_views():
    slicer.util.resetSliceViews()


def initialize_cases_from_csv():
    global cases, current_case_index, current_case_uid

    rows = _read_csv_rows(CSV_PATH)
    cases = _build_cases_from_long_rows(rows)

    def _safe_pair_index(case):
        try:
            return int(case["pair_index"])
        except Exception:
            return 999999

    cases.sort(key=lambda x: (x["patient_id"], _safe_pair_index(x)))
    current_case_index = -1
    current_case_uid = None

    print(f"CSV leído: {CSV_PATH}")
    print(f"Casos encontrados: {len(cases)}")

def _read_outcomes():
    outcomes = {}

    if not os.path.exists(OUTCOME_CSV):
        return outcomes

    with open(OUTCOME_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("uid")
            if uid:
                outcomes[uid] = row

    return outcomes


def _write_outcomes(outcomes_dict):
    os.makedirs(os.path.dirname(OUTCOME_CSV), exist_ok=True)

    with open(OUTCOME_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["uid", "patient_id", "pair_index", "outcome"]
        )
        writer.writeheader()

        for uid in sorted(outcomes_dict.keys()):
            writer.writerow(outcomes_dict[uid])


def get_existing_outcome(uid=None):
    global current_case_uid

    if uid is None:
        uid = current_case_uid

    if uid is None:
        return None

    outcomes = _read_outcomes()
    row = outcomes.get(uid)
    if row is None:
        return None

    return row.get("outcome")


def classify_current_case(outcome):
    global current_case_uid, current_case_index

    if current_case_uid is None or current_case_index < 0:
        return

    outcome_str = str(outcome).strip()

    case = cases[current_case_index]
    outcomes = _read_outcomes()

    outcomes[current_case_uid] = {
        "uid": current_case_uid,
        "patient_id": case["patient_id"],
        "pair_index": case["pair_index"],
        "outcome": outcome_str,
    }

    _write_outcomes(outcomes)
    print(f"Clasificación guardada: uid={current_case_uid}, outcome={outcome_str}")
    update_reviewer_status()


def show_classification_status():
    outcomes = _read_outcomes()
    total = len(cases)
    classified = 0

    for case in cases:
        if case["uid"] in outcomes:
            classified += 1

    msg = (
        f"Casos totales: {total}\n"
        f"Casos clasificados: {classified}\n"
        f"Casos pendientes: {max(total - classified, 0)}"
    )
    return msg


def load_case(index):
    global current_case_index, current_case_uid

    if index < 0 or index >= len(cases):
        print(f"Fuera de rango: {index}")
        return

    case = cases[index]
    current_case_index = index
    current_case_uid = case["uid"]

    print(f"\nCargando caso {index + 1}/{len(cases)}")
    print(f"  patient_id: {case['patient_id']}")
    print(f"  pair_index: {case['pair_index']}")
    print(f"  uid: {case['uid']}")

    slicer.mrmlScene.Clear()
    _set_two_slice_layout()

    vol_base = _load_volume(case["baseline_image"])
    vol_fu = _load_volume(case["followup_image"])
    seg_base = _load_segmentation(case["baseline_mask"])
    seg_fu = _load_segmentation(case["followup_mask"])

    _configure_slice_view("Red", vol_base, orientation="Axial")
    _configure_slice_view("Yellow", vol_fu, orientation="Axial")

    _show_segmentation_only_in_view(seg_base, "Red")
    _show_segmentation_only_in_view(seg_fu, "Yellow")

    _fit_views()

    print("Caso cargado correctamente.")
    print(f"  Red    -> {os.path.basename(case['baseline_image'])} + {os.path.basename(case['baseline_mask'])}")
    print(f"  Yellow -> {os.path.basename(case['followup_image'])} + {os.path.basename(case['followup_mask'])}")

    update_reviewer_status()


def load_next_case():
    global current_case_index
    next_index = current_case_index + 1
    if next_index >= len(cases):
        return
    load_case(next_index)


def load_prev_case():
    global current_case_index
    prev_index = current_case_index - 1
    if prev_index < 0:
        return
    load_case(prev_index)


def load_next_unclassified():
    outcomes = _read_outcomes()

    for i, case in enumerate(cases):
        if case["uid"] not in outcomes:
            load_case(i)
            return

    slicer.util.infoDisplay("No quedan casos sin clasificar")


def update_reviewer_status():
    global reviewer_widget, current_case_index

    if reviewer_widget is None:
        return

    total = len(cases)
    if current_case_index < 0 or current_case_index >= total:
        reviewer_widget.caseLabel.text = "Ningun caso cargado"
        reviewer_widget.outcomeLabel.text = "Outcome actual: -"
        reviewer_widget.progressLabel.text = show_classification_status()
        return

    case = cases[current_case_index]
    existing = get_existing_outcome()

    reviewer_widget.caseLabel.text = (
        f"Caso {current_case_index + 1}/{total}\n"
        f"patient_id: {case['patient_id']}\n"
        f"pair_index: {case['pair_index']}\n"
        f"uid: {case['uid']}"
    )

    reviewer_widget.outcomeLabel.text = f"Outcome actual: {existing if existing is not None else 'sin clasificar'}"
    reviewer_widget.progressLabel.text = show_classification_status()


class CaseReviewerWidget(qt.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Case Comparison Reviewer")
        self.setMinimumWidth(360)

        layout = qt.QVBoxLayout()
        self.setLayout(layout)

        self.csvLabel = qt.QLabel(f"CSV: {CSV_PATH}")
        self.csvLabel.wordWrap = True
        layout.addWidget(self.csvLabel)

        self.caseLabel = qt.QLabel("Ningún caso cargado")
        self.caseLabel.wordWrap = True
        layout.addWidget(self.caseLabel)

        self.outcomeLabel = qt.QLabel("Outcome actual: -")
        layout.addWidget(self.outcomeLabel)

        self.progressLabel = qt.QLabel("")
        self.progressLabel.wordWrap = True
        layout.addWidget(self.progressLabel)

        layout.addSpacing(8)

        self.loadButton = qt.QPushButton("Inicializar CSV")
        self.loadButton.clicked.connect(self.onInitialize)
        layout.addWidget(self.loadButton)

        self.firstUnclassifiedButton = qt.QPushButton("Cargar siguiente sin clasificar")
        self.firstUnclassifiedButton.clicked.connect(self.onNextUnclassified)
        layout.addWidget(self.firstUnclassifiedButton)

        navigationLayout = qt.QHBoxLayout()
        self.prevButton = qt.QPushButton("Anterior")
        self.prevButton.clicked.connect(self.onPrev)
        navigationLayout.addWidget(self.prevButton)

        self.nextButton = qt.QPushButton("Siguiente")
        self.nextButton.clicked.connect(self.onNext)
        navigationLayout.addWidget(self.nextButton)
        layout.addLayout(navigationLayout)

        layout.addSpacing(8)

        classNextLayout = qt.QHBoxLayout()
        self.class0NextButton = qt.QPushButton("No mejora")
        self.class0NextButton.clicked.connect(self.onClass0Next)
        classNextLayout.addWidget(self.class0NextButton)

        self.class1NextButton = qt.QPushButton("Mejora")
        self.class1NextButton.clicked.connect(self.onClass1Next)
        classNextLayout.addWidget(self.class1NextButton)
        layout.addLayout(classNextLayout)

        layout.addSpacing(8)

        update_reviewer_status()

    def onInitialize(self):
        try:
            initialize_cases_from_csv()
            if len(cases) > 0:
                load_case(0)
            else:
                slicer.util.infoDisplay("No se han encontrado casos válidos en el CSV.")
                update_reviewer_status()
        except Exception as e:
            slicer.util.errorDisplay(f"Error al inicializar casos:\n{e}")

    def onNextUnclassified(self):
        try:
            load_next_unclassified()
        except Exception as e:
            slicer.util.errorDisplay(f"Error al cargar siguiente no clasificado:\n{e}")

    def onPrev(self):
        try:
            load_prev_case()
        except Exception as e:
            slicer.util.errorDisplay(f"Error al cargar caso anterior:\n{e}")

    def onNext(self):
        try:
            load_next_case()
        except Exception as e:
            slicer.util.errorDisplay(f"Error al cargar siguiente caso:\n{e}")

    def onClass0Next(self):
        try:
            classify_current_case(0)
            load_next_case()
        except Exception as e:
            slicer.util.errorDisplay(f"Error al clasificar y avanzar:\n{e}")

    def onClass1Next(self):
        try:
            classify_current_case(1)
            load_next_case()
        except Exception as e:
            slicer.util.errorDisplay(f"Error al clasificar y avanzar:\n{e}")

def launch_case_reviewer():
    global reviewer_widget

    if reviewer_widget is not None:
        reviewer_widget.close()
        reviewer_widget = None

    reviewer_widget = CaseReviewerWidget()
    reviewer_widget.show()
    reviewer_widget.raise_()
    reviewer_widget.activateWindow()
    update_reviewer_status()
