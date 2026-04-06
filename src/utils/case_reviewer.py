import os
import csv
import slicer
import qt

PROJECT_ROOT = r"C:\Users\PabolimorWS\Desktop\TFG"
CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "pairs", "longitudinal_pairs.csv")

OUTCOME_CSV = os.path.join(PROJECT_ROOT, "outputs", "review", "classification_outcomes.csv")

REQUIRE_RADIOMICS_VALID = True
REQUIRE_COMPLETE_PAIR = True


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


def _case_uid(patient_id, pair_index):
    return f"{patient_id}__pair_{pair_index}"


def _load_volume(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el volumen: {path}")
    node = slicer.util.loadVolume(path)
    if node is None:
        raise RuntimeError(f"No se pudo cargar volumen: {path}")
    return node


def _load_segmentation(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe la segmentación: {path}")
    node = slicer.util.loadSegmentation(path)
    if node is None:
        raise RuntimeError(f"No se pudo cargar segmentación: {path}")
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
    if display_node is None:
        raise RuntimeError(f"La segmentación {segmentation_node.GetName()} no tiene display node")

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
    cases = []

    for row in rows:
        patient_id = str(row.get("patient_id", "")).strip()
        pair_index = str(row.get("pair_index", "")).strip()
        uid = str(row.get("uid", "")).strip()

        if patient_id == "":
            continue

        if pair_index == "":
            pair_index = "0"

        if uid == "":
            uid = _case_uid(patient_id, pair_index)

        if REQUIRE_RADIOMICS_VALID and "is_radiomics_valid" in row:
            if not _parse_bool(row["is_radiomics_valid"]):
                continue

        baseline_image = _resolve_path(
            row.get("baseline_image_path") or row.get("baseline_image")
        )
        baseline_mask = _resolve_path(
            row.get("baseline_mask_path") or row.get("baseline_mask")
        )
        followup_image = _resolve_path(
            row.get("followup_image_path") or row.get("followup_image")
        )
        followup_mask = _resolve_path(
            row.get("followup_mask_path") or row.get("followup_mask")
        )

        has_baseline = baseline_image and baseline_mask
        has_followup = followup_image and followup_mask

        if REQUIRE_COMPLETE_PAIR and not (has_baseline and has_followup):
            continue

        if not has_baseline or not has_followup:
            continue

        cases.append({
            "uid": uid,
            "patient_id": patient_id,
            "pair_index": pair_index,
            "baseline_image": baseline_image,
            "baseline_mask": baseline_mask,
            "followup_image": followup_image,
            "followup_mask": followup_mask,
            "row": row,
        })

    def _safe_pair_index(case):
        try:
            return int(case["pair_index"])
        except Exception:
            return 999999

    cases.sort(key=lambda x: (x["patient_id"], _safe_pair_index(x)))
    current_case_index = -1
    current_case_uid = None

    print(f"CSV leído: {CSV_PATH}")
    print(f"Casos longitudinales encontrados: {len(cases)}")

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
        slicer.util.infoDisplay("No hay ningún caso cargado.")
        return

    outcome_str = str(outcome).strip()
    if outcome_str not in ("0", "1"):
        raise ValueError("El outcome debe ser 0 o 1.")

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
    print(msg)
    return msg


def load_case(index):
    global current_case_index, current_case_uid

    if not cases:
        raise RuntimeError("Primero ejecuta initialize_cases_from_csv().")

    if index < 0 or index >= len(cases):
        print(f"Índice fuera de rango: {index}")
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
        slicer.util.infoDisplay("Ya estás en el último caso.")
        return
    load_case(next_index)


def load_prev_case():
    global current_case_index
    prev_index = current_case_index - 1
    if prev_index < 0:
        slicer.util.infoDisplay("Ya estás en el primer caso.")
        return
    load_case(prev_index)


def load_next_unclassified():
    outcomes = _read_outcomes()

    for i, case in enumerate(cases):
        if case["uid"] not in outcomes:
            load_case(i)
            return

    slicer.util.infoDisplay("No quedan casos sin clasificar.")


def update_reviewer_status():
    global reviewer_widget, current_case_index

    if reviewer_widget is None:
        return

    total = len(cases)
    if current_case_index < 0 or current_case_index >= total:
        reviewer_widget.caseLabel.text = "Ningún caso cargado"
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
        self.setWindowTitle("TFG Case Reviewer")
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

        self.refreshButton = qt.QPushButton("Actualizar estado")
        self.refreshButton.clicked.connect(self.onRefresh)
        layout.addWidget(self.refreshButton)

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

    def onClass0(self):
        try:
            classify_current_case(0)
        except Exception as e:
            slicer.util.errorDisplay(f"Error al clasificar:\n{e}")

    def onClass1(self):
        try:
            classify_current_case(1)
        except Exception as e:
            slicer.util.errorDisplay(f"Error al clasificar:\n{e}")

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

    def onRefresh(self):
        update_reviewer_status()


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
