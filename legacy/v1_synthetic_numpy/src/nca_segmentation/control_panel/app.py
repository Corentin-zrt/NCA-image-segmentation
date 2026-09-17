from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from nca_segmentation.config import DATASET_DIR, MODEL_PATH, DatasetConfig
from nca_segmentation.data.generator import Dataset, generate_dataset
from nca_segmentation.data.io import save_dataset
from nca_segmentation.control_panel.visualization import label_colors, segmentation_metrics
from nca_segmentation.nca.model import NeuralCellularAutomaton


class ControlPanel:
    WIDTH = 1180
    HEIGHT = 720
    BG = (12, 18, 28)
    PANEL = (22, 31, 45)
    PANEL_ALT = (29, 40, 56)
    TEXT = (232, 238, 246)
    MUTED = (142, 157, 177)
    ACCENT = (53, 211, 153)
    ORANGE = (255, 174, 92)
    DANGER = (238, 103, 110)

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("NCA / IMAGE SEGMENTATION")
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 34)
        self.config = DatasetConfig()
        self.dataset: Dataset | None = None
        self.prediction_mask: np.ndarray | None = None
        self.training_step = 0
        self.training_active = False
        self.training_loss: float | None = None
        self.model = NeuralCellularAutomaton()
        self.preview_index = 0
        self.shape_options = ["multi", "mixed", "circle", "square", "triangle"]
        self.shape_index = 0
        self.status = "Ready. Generate a dataset to begin."
        self.running = True

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                self._handle_event(event)
            if self.training_active:
                self._train_one_step()
            self._draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.running = False
        if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
            return

        position = event.pos
        if self._button("shape", position):
            self.shape_index = (self.shape_index + 1) % len(self.shape_options)
            self.status = f"Shape set to {self.shape_options[self.shape_index]}."
        elif self._button("generate", position):
            self._generate()
        elif self._button("preview", position):
            self._next_preview()
        elif self._button("save", position):
            self._save()
        elif self._button("save_model", position):
            self._save_model()
        elif self._button("load_model", position):
            self._load_model()
        elif self._button("train", position):
            self._toggle_training()
        elif self._button("quit", position):
            self.running = False

    def _generate(self) -> None:
        self.config.shape = self.shape_options[self.shape_index]
        self.dataset = generate_dataset(
            samples=self.config.samples,
            image_size=self.config.image_size,
            shape=self.config.shape,
            seed=self.config.seed,
        )
        self.preview_index = 0
        self.prediction_mask = None
        self.training_step = 0
        self.training_loss = None
        self.training_active = False
        self.status = f"Generated {self.config.samples} {self.config.shape} samples."

    def _toggle_training(self) -> None:
        if self.dataset is None:
            self.status = "Generate a dataset before training the NCA."
            return
        self.training_active = not self.training_active
        if self.training_active:
            self.status = "Training started. Watch the prediction evolve."
        else:
            self.status = "Training paused."

    def _train_one_step(self) -> None:
        assert self.dataset is not None
        self.training_loss, _ = self.model.train_step(
            self.dataset.images,
            self.dataset.masks,
        )
        prediction = self.model.forward(self.dataset.images[self.preview_index])
        self.update_prediction(prediction, self.training_step + 1)
        self.status = f"Training step {self.training_step} | loss {self.training_loss:.4f}"

    def update_prediction(self, mask: np.ndarray, step: int = 0) -> None:
        """Update the visible prediction from a training or inference callback."""
        if self.dataset is None:
            raise RuntimeError("Generate a dataset before displaying a prediction")
        expected_shape = self.dataset.masks[self.preview_index].shape
        if mask.shape != expected_shape:
            raise ValueError(f"prediction must have shape {expected_shape}")
        prediction = mask.astype(np.int64)
        if prediction.min() < 0 or prediction.max() >= len(self.dataset.class_names):
            raise ValueError("prediction contains an unknown class label")
        self.prediction_mask = prediction.astype(np.uint8)
        self.training_step = step

    def _save(self) -> None:
        if self.dataset is None:
            self.status = "Generate a dataset before saving it."
            return
        path = save_dataset(self.dataset, DATASET_DIR)
        self.status = f"Saved dataset to {path.parent}."

    def _save_model(self) -> None:
        if self.dataset is None or self.training_step == 0:
            self.status = "Train the NCA before saving its model."
            return
        path = self.model.save(MODEL_PATH)
        self.status = f"Saved model to {path}."

    def _load_model(self) -> None:
        if self.dataset is None:
            self.status = "Generate a dataset before loading a model."
            return
        if not MODEL_PATH.exists():
            self.status = f"No saved model found at {MODEL_PATH}."
            return
        loaded_model = NeuralCellularAutomaton.load(MODEL_PATH)
        expected_classes = len(self.dataset.class_names)
        if loaded_model.config.classes != expected_classes:
            self.status = f"Model has {loaded_model.config.classes} classes; dataset needs {expected_classes}."
            return
        self.model = loaded_model
        self.prediction_mask = self.model.forward(self.dataset.images[self.preview_index])
        self.training_loss = None
        self.training_active = False
        self.status = f"Loaded model from {MODEL_PATH}."

    def _next_preview(self) -> None:
        if self.dataset is None:
            self.status = "Generate a dataset before previewing it."
            return
        self.preview_index = (self.preview_index + 1) % len(self.dataset.images)
        self.prediction_mask = self.model.forward(self.dataset.images[self.preview_index])
        self.status = f"Preview {self.preview_index + 1}/{len(self.dataset.images)}."

    def _button(self, name: str, position: tuple[int, int]) -> bool:
        rectangles = {
            "shape": pygame.Rect(40, 170, 270, 44),
            "generate": pygame.Rect(40, 246, 270, 48),
            "preview": pygame.Rect(40, 306, 130, 42),
            "save": pygame.Rect(180, 306, 130, 42),
            "train": pygame.Rect(40, 388, 270, 48),
            "save_model": pygame.Rect(180, 388, 130, 48),
            "load_model": pygame.Rect(40, 452, 270, 42),
            "quit": pygame.Rect(40, 610, 270, 42),
        }
        return rectangles[name].collidepoint(position)

    def _draw(self) -> None:
        self.screen.fill(self.BG)
        self._draw_sidebar()
        self._draw_workspace()
        self._draw_status()

    def _draw_sidebar(self) -> None:
        pygame.draw.rect(self.screen, self.PANEL, (0, 0, 350, self.HEIGHT))
        self._text("NCA / SEGMENTATION", (40, 40), self.title_font, self.ACCENT)
        self._text("CONTROL PANEL", (40, 78), self.small_font, self.MUTED)
        self._text("DATASET SHAPE", (40, 135), self.small_font, self.MUTED)
        shape = self.shape_options[self.shape_index].upper()
        self._button_draw(pygame.Rect(40, 170, 270, 44), f"SHAPE  /  {shape}", self.ORANGE)
        self._text("DATASET", (40, 226), self.small_font, self.MUTED)
        self._button_draw(pygame.Rect(40, 246, 270, 48), "GENERATE DATASET", self.ACCENT)
        self._button_draw(pygame.Rect(40, 306, 130, 42), "PREVIEW NEXT", self.PANEL_ALT)
        self._button_draw(pygame.Rect(180, 306, 130, 42), "SAVE DATA", self.PANEL_ALT)
        self._text("NCA PIPELINE", (40, 368), self.small_font, self.MUTED)
        train_label = "PAUSE TRAINING" if self.training_active else "TRAIN NCA"
        self._button_draw(pygame.Rect(40, 388, 130, 48), train_label, self.DANGER)
        self._button_draw(pygame.Rect(180, 388, 130, 48), "SAVE MODEL", self.PANEL_ALT)
        self._button_draw(pygame.Rect(40, 452, 270, 42), "LOAD MODEL", self.PANEL_ALT)
        if self.training_loss is not None:
            self._text(f"LOSS  {self.training_loss:.4f}", (40, 506), self.small_font, self.ACCENT)
        self._text("ESC  close panel", (40, 580), self.small_font, self.MUTED)
        self._button_draw(pygame.Rect(40, 610, 270, 42), "QUIT", self.PANEL_ALT)

    def _draw_workspace(self) -> None:
        self._text("DATASET PREVIEW", (390, 42), self.title_font, self.TEXT)
        self._text("PREDICTION / TARGET", (390, 81), self.small_font, self.MUTED)
        left = pygame.Rect(390, 125, 230, 350)
        middle = pygame.Rect(645, 125, 230, 350)
        right = pygame.Rect(900, 125, 230, 350)
        for rectangle in (left, middle, right):
            pygame.draw.rect(self.screen, self.PANEL, rectangle)
        if self.dataset is None:
            self._empty_preview(left, "NO IMAGE")
            self._empty_preview(middle, "NO PREDICTION")
            self._empty_preview(right, "NO MASK")
            return
        image = self.dataset.images[self.preview_index]
        mask = self.dataset.masks[self.preview_index]
        self._blit_array(image, left)
        if self.prediction_mask is None:
            self._empty_preview(middle, "WAITING")
        else:
            prediction = label_colors(self.prediction_mask)
            self._blit_array(prediction, middle)
        mask_rgb = label_colors(mask)
        self._blit_array(mask_rgb, right)
        self._text("IMAGE   /   PREDICTION   /   TARGET", (390, 500), self.font, self.TEXT)
        metrics = segmentation_metrics(self.prediction_mask, mask) if self.prediction_mask is not None else {"iou": 0.0, "coverage": 0.0}
        self._text(
            f"MEAN {metrics['iou']:.1%}  C {metrics.get('iou_1', 0):.1%}  S {metrics.get('iou_2', 0):.1%}  T {metrics.get('iou_3', 0):.1%}",
            (390, 530),
            self.small_font,
            self.MUTED,
        )
        self._text(f"SAMPLE {self.preview_index + 1:02d}/{len(self.dataset.images):02d}  |  STEP {self.training_step}", (780, 530), self.small_font, self.MUTED)
        self._text("RED circle   BLUE square   GOLD triangle", (390, 560), self.small_font, self.MUTED)

    def _draw_status(self) -> None:
        pygame.draw.rect(self.screen, self.PANEL_ALT, (390, 605, 740, 48))
        self._text("STATUS", (408, 620), self.small_font, self.ACCENT)
        self._text(self.status, (485, 620), self.small_font, self.TEXT)

    def _empty_preview(self, rectangle: pygame.Rect, label: str) -> None:
        self._text(label, (rectangle.centerx - 35, rectangle.centery - 8), self.font, self.MUTED)

    def _blit_array(self, array: np.ndarray, rectangle: pygame.Rect) -> None:
        surface = pygame.surfarray.make_surface(np.transpose(array, (1, 0, 2)))
        surface = pygame.transform.smoothscale(surface, rectangle.size)
        self.screen.blit(surface, rectangle)

    def _button_draw(self, rectangle: pygame.Rect, label: str, color: tuple[int, int, int]) -> None:
        pygame.draw.rect(self.screen, color, rectangle, border_radius=4)
        text_color = self.BG if color != self.PANEL_ALT else self.TEXT
        text_surface = self.small_font.render(label, True, text_color)
        self.screen.blit(text_surface, text_surface.get_rect(center=rectangle.center))

    def _text(self, value: str, position: tuple[int, int], font: pygame.font.Font, color: tuple[int, int, int]) -> None:
        self.screen.blit(font.render(value, True, color), position)
