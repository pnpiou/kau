import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import os
from datetime import datetime
import matplotlib.pyplot as plt

class RayVisualizer:
    def __init__(self, width=40, height=40):
        self.width = width
        self.height = height
        # 0.5cm마다 색상, 최대 200cm까지 400단계 (더 극단적 그라데이션)
        self.colors = []
        for i in range(400):
            hue = i / 400  # 0~1
            rgb = self.hsv_to_rgb(hue, 1.0, 1.0)
            self.colors.append(tuple(int(x * 255) for x in rgb))
        self.data_dir = os.path.join(os.path.expanduser("~"), "Desktop", "실시간 창")
        os.makedirs(self.data_dir, exist_ok=True)
        self.last_save_time = datetime.now()
        self.save_buffer = []
        self.save_file = os.path.join(self.data_dir, "ray_vectors_log.txt")
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))
        plt.ion()
        plt.show()

    def hsv_to_rgb(self, h, s, v):
        if s == 0.0:
            return (v, v, v)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0: return (v, t, p)
        if i == 1: return (q, v, p)
        if i == 2: return (p, v, t)
        if i == 3: return (p, q, v)
        if i == 4: return (t, p, v)
        if i == 5: return (v, p, q)

    def update_visualization(self, ray_data):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        line = []
        for i, (distance, vector) in enumerate(ray_data):
            x = i % self.width
            y = i // self.width
            if distance == -1:
                color = (128, 128, 128)
            else:
                color_idx = int(distance / 0.5)
                color_idx = min(max(color_idx, 0), len(self.colors) - 1)
                color = self.colors[color_idx]
            img[y, x] = color
            line.append(f"{i}:({vector[0]:.2f},{vector[1]:.2f},{vector[2]:.2f})")
        # 오른쪽 90도 회전 + 상하좌우 반전
        img_rot = np.rot90(img, k=-1)
        img_flip = np.flipud(np.fliplr(img_rot))
        self.im.set_data(img_flip)
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        self.frame_count += 1
        if self.frame_count % 3 == 0:
            plt.draw()
            plt.pause(0.001)
        self.save_buffer.append(" ".join(line))
        now = datetime.now()
        if (now - self.last_save_time).total_seconds() >= 1/30:
            self.save_data()
            self.last_save_time = now

    def save_data(self):
        if not self.save_buffer:
            return
        with open(self.save_file, "a", encoding="utf-8") as f:
            for line in self.save_buffer:
                f.write(line + "\n")
        self.save_buffer = []

    def close(self):
        plt.close(self.fig)

class ToFSimulator:
    def __init__(self):
        # Pygame 초기화
        pygame.init()
        
        # 디스플레이 설정
        self.display = (800, 800)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("ToF 센서 시뮬레이션")
        
        # GLUT 초기화 제거 (폰트 렌더링만 필요하면 glutInit()만 남김)
        # glutInit()
        # glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        
        # OpenGL 컨텍스트 설정
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        
        # 기본 상태 설정
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        
        # 조명 설정
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 200, 0, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1))
        
        # 카메라 설정
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.display[0]/self.display[1]), 0.1, 10000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 200, 400,  # 카메라 위치
                  0, 0, 0,      # 바라보는 점
                  0, 1, 0)      # 카메라의 위쪽 방향

        # 레이캐스팅 시뮬레이션 창 생성
        self.ray_visualizer = RayVisualizer(width=40, height=40)

        # 센서 관련 변수 초기화
        self.sensor_pos = [0, 10, 0]
        self.sensor_rotation = [0, 0, 0]
        self.space_rotation = [0, 0, 0]
        self.show_cone = True
        self.space_translation = [0, 0, 0]  # 공간 전체 이동 (x, y, z)

        # 상수 정의
        self.DETECTION_RANGE = 200  # 원뿔 길이 (cm)
        self.DETECTION_ANGLE = 30   # 원뿔 각도 (도)
        self.RAY_RESOLUTION = 3     # 3cm당 하나의 ray
        self.MAX_RAYS = 100         # 최대 ray 개수

        # 테스트용 물체
        self.test_objects = [
            {'type': 'sphere', 'pos': [100, 10, 100], 'size': 40},
            {'type': 'cube', 'pos': [-100, 10, -100], 'size': 40},
            {'type': 'rect', 'pos': [0, 10, 200], 'size': 40, 'size_y': 80, 'size_z': 80},
            {'type': 'cylinder', 'pos': [200, 20, 0], 'radius': 20, 'height': 60},
            {'type': 'tetrahedron', 'pos': [-200, 30, 0], 'size': 40}
        ]

        # 상태 변수
        self.running = True
        self.dragging = False
        self.is_rotating = False
        self.last_mouse_pos = None

    def draw_grid(self):
        glBegin(GL_LINES)
        glColor3f(0.3, 0.3, 0.3)
        for i in range(-800, 801, 50):
            glVertex3f(i, 0, -800)
            glVertex3f(i, 0, 800)
            glVertex3f(-800, 0, i)
            glVertex3f(800, 0, i)
        glEnd()

    def draw_room(self):
        # 바닥
        glColor3f(0.2, 0.2, 0.2)
        glBegin(GL_QUADS)
        glVertex3f(-800, 0, -800)
        glVertex3f(800, 0, -800)
        glVertex3f(800, 0, 800)
        glVertex3f(-800, 0, 800)
        glEnd()

        # 천장
        glColor3f(0.15, 0.15, 0.15)
        glBegin(GL_QUADS)
        glVertex3f(-800, 800, -800)
        glVertex3f(800, 800, -800)
        glVertex3f(800, 800, 800)
        glVertex3f(-800, 800, 800)
        glEnd()

        # 벽 4개
        glColor3f(0.25, 0.25, 0.3)
        # 앞
        glBegin(GL_QUADS)
        glVertex3f(-800, 0, -800)
        glVertex3f(800, 0, -800)
        glVertex3f(800, 800, -800)
        glVertex3f(-800, 800, -800)
        glEnd()
        # 뒤
        glBegin(GL_QUADS)
        glVertex3f(-800, 0, 800)
        glVertex3f(800, 0, 800)
        glVertex3f(800, 800, 800)
        glVertex3f(-800, 800, 800)
        glEnd()
        # 왼쪽
        glBegin(GL_QUADS)
        glVertex3f(-800, 0, -800)
        glVertex3f(-800, 0, 800)
        glVertex3f(-800, 800, 800)
        glVertex3f(-800, 800, -800)
        glEnd()
        # 오른쪽
        glBegin(GL_QUADS)
        glVertex3f(800, 0, -800)
        glVertex3f(800, 0, 800)
        glVertex3f(800, 800, 800)
        glVertex3f(800, 800, -800)
        glEnd()

    def draw_cube(self, size):
        hs = size / 2
        glBegin(GL_QUADS)
        # 위
        glVertex3f(-hs, hs, -hs)
        glVertex3f(hs, hs, -hs)
        glVertex3f(hs, hs, hs)
        glVertex3f(-hs, hs, hs)
        # 아래
        glVertex3f(-hs, -hs, -hs)
        glVertex3f(hs, -hs, -hs)
        glVertex3f(hs, -hs, hs)
        glVertex3f(-hs, -hs, hs)
        # 앞
        glVertex3f(-hs, -hs, -hs)
        glVertex3f(hs, -hs, -hs)
        glVertex3f(hs, hs, -hs)
        glVertex3f(-hs, hs, -hs)
        # 뒤
        glVertex3f(-hs, -hs, hs)
        glVertex3f(hs, -hs, hs)
        glVertex3f(hs, hs, hs)
        glVertex3f(-hs, hs, hs)
        # 왼쪽
        glVertex3f(-hs, -hs, -hs)
        glVertex3f(-hs, -hs, hs)
        glVertex3f(-hs, hs, hs)
        glVertex3f(-hs, hs, -hs)
        # 오른쪽
        glVertex3f(hs, -hs, -hs)
        glVertex3f(hs, -hs, hs)
        glVertex3f(hs, hs, hs)
        glVertex3f(hs, hs, -hs)
        glEnd()

    def draw_rect_box(self, size_x, size_y, size_z):
        hx, hy, hz = size_x/2, size_y/2, size_z/2
        glBegin(GL_QUADS)
        # 위
        glVertex3f(-hx, hy, -hz)
        glVertex3f(hx, hy, -hz)
        glVertex3f(hx, hy, hz)
        glVertex3f(-hx, hy, hz)
        # 아래
        glVertex3f(-hx, -hy, -hz)
        glVertex3f(hx, -hy, -hz)
        glVertex3f(hx, -hy, hz)
        glVertex3f(-hx, -hy, hz)
        # 앞
        glVertex3f(-hx, -hy, -hz)
        glVertex3f(hx, -hy, -hz)
        glVertex3f(hx, hy, -hz)
        glVertex3f(-hx, hy, -hz)
        # 뒤
        glVertex3f(-hx, -hy, hz)
        glVertex3f(hx, -hy, hz)
        glVertex3f(hx, hy, hz)
        glVertex3f(-hx, hy, hz)
        # 왼쪽
        glVertex3f(-hx, -hy, -hz)
        glVertex3f(-hx, -hy, hz)
        glVertex3f(-hx, hy, hz)
        glVertex3f(-hx, hy, -hz)
        # 오른쪽
        glVertex3f(hx, -hy, -hz)
        glVertex3f(hx, -hy, hz)
        glVertex3f(hx, hy, hz)
        glVertex3f(hx, hy, -hz)
        glEnd()

    def draw_cone(self, sensor_y=0):
        glPushMatrix()
        glTranslatef(0, 0, 0)
        glRotatef(90, 1, 0, 0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.0, 1.0, 0.0, 0.2)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        
        max_length = self.DETECTION_RANGE
        if sensor_y > 0:
            h = sensor_y
            theta = math.radians(self.DETECTION_ANGLE)
            max_length = min(self.DETECTION_RANGE, h / math.cos(theta))
            
            for i in range(37):
                angle = i * 10 * math.pi / 180
                x = max_length * math.cos(angle) * math.sin(theta)
                y = max_length * math.sin(angle) * math.sin(theta)
                z = max_length * math.cos(theta)
                if y < -h:
                    y = -h
                glVertex3f(x, y, z)
        else:
            for i in range(37):
                angle = i * 10 * math.pi / 180
                x = self.DETECTION_RANGE * math.cos(angle)
                y = self.DETECTION_RANGE * math.sin(angle)
                if y < 0:
                    y = 0
                glVertex3f(x, y, self.DETECTION_RANGE)
        glEnd()
        glDisable(GL_BLEND)
        glPopMatrix()

    def draw_test_objects(self):
        for obj in self.test_objects:
            glPushMatrix()
            if obj['type'] == 'sphere':
                y = max(obj['pos'][1], obj['size']/2)
                glTranslatef(obj['pos'][0], y, obj['pos'][2])
                glColor3f(0.5, 0.5, 0.5)
                sphere = gluNewQuadric()
                gluSphere(sphere, obj['size']/2, 32, 32)
            elif obj['type'] == 'cube':
                y = max(obj['pos'][1], obj['size']/2)
                glTranslatef(obj['pos'][0], y, obj['pos'][2])
                glColor3f(0.7, 0.4, 0.4)
                self.draw_cube(obj['size'])
            elif obj['type'] == 'rect':
                y = max(obj['pos'][1], obj.get('size_y', obj['size'])/2)
                glTranslatef(obj['pos'][0], y, obj['pos'][2])
                glColor3f(0.4, 0.4, 0.7)
                self.draw_rect_box(obj['size'], obj.get('size_y', obj['size']), obj.get('size_z', obj['size']))
            elif obj['type'] == 'cylinder':
                y = max(obj['pos'][1], obj['height']/2)
                glTranslatef(obj['pos'][0], y, obj['pos'][2])
                glColor3f(0.2, 0.7, 0.2)
                quad = gluNewQuadric()
                gluCylinder(quad, obj['radius'], obj['radius'], obj['height'], 32, 1)
            elif obj['type'] == 'tetrahedron':
                y = obj['pos'][1]
                glTranslatef(obj['pos'][0], y, obj['pos'][2])
                glColor3f(0.8, 0.8, 0.2)
                # 정사면체 꼭짓점 계산 (중심 기준, edge 길이)
                a = obj['size']
                h = math.sqrt(2/3) * a
                v0 = np.array([0, h/2, 0])
                v1 = np.array([a/2, -h/2, -a/(2*math.sqrt(3))])
                v2 = np.array([-a/2, -h/2, -a/(2*math.sqrt(3))])
                v3 = np.array([0, -h/2, a/math.sqrt(3)])
                glBegin(GL_TRIANGLES)
                # 4면
                for tri in [(v0, v1, v2), (v0, v2, v3), (v0, v3, v1), (v1, v2, v3)]:
                    for v in tri:
                        glVertex3f(*v)
                glEnd()
            glPopMatrix()

    def draw_direction_arrow(self):
        glPushMatrix()
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, -30)
        glVertex3f(0, 0, -30)
        glVertex3f(-5, 0, -25)
        glVertex3f(0, 0, -30)
        glVertex3f(5, 0, -25)
        glEnd()
        glPopMatrix()

    def draw_tof_sensor(self):
        glPushMatrix()
        y = max(self.sensor_pos[1], 10)
        glTranslatef(self.sensor_pos[0], y, self.sensor_pos[2])
        glRotatef(self.sensor_rotation[0], 1, 0, 0)
        glRotatef(self.sensor_rotation[1], 0, 1, 0)
        glRotatef(self.sensor_rotation[2], 0, 0, 1)
        glColor3f(0.0, 1.0, 0.0)
        sphere = gluNewQuadric()
        gluSphere(sphere, 10, 32, 32)
        glColor3f(1.0, 1.0, 1.0)
        glRasterPos3f(-5, 15, 0)
        for char in "ToF":
            pass
        # 레이캐스팅 범위 시각화: 각 ray 방향으로 선 그리기
        cone_radius = self.DETECTION_RANGE * math.tan(math.radians(self.DETECTION_ANGLE))
        num_rays = 40
        glColor3f(1.0, 0.0, 1.0)
        glBegin(GL_LINES)
        for i in range(num_rays):
            for j in range(num_rays):
                x = -cone_radius + i * (cone_radius * 2) / (num_rays - 1)
                y_ray = -cone_radius + j * (cone_radius * 2) / (num_rays - 1)
                if x**2 + y_ray**2 > cone_radius**2:
                    continue
                ray_dir = np.array([x, y_ray, -self.DETECTION_RANGE])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                end_point = ray_dir * self.DETECTION_RANGE
                glVertex3f(0, 0, 0)
                glVertex3f(end_point[0], end_point[1], end_point[2])
        glEnd()
        self.draw_direction_arrow()
        glPopMatrix()

    def calculate_ray_hits(self):
        ray_data = []
        cone_radius = self.DETECTION_RANGE * math.tan(math.radians(self.DETECTION_ANGLE))
        num_rays = 40
        sensor_pos = np.array(self.sensor_pos)
        # 회전 행렬 계산 (pitch, yaw)
        pitch = math.radians(self.sensor_rotation[0])
        yaw = math.radians(self.sensor_rotation[1])
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]
        ])
        Ry = np.array([
            [math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)]
        ])
        R = Ry @ Rx
        for i in range(num_rays):
            for j in range(num_rays):
                x = -cone_radius + i * (cone_radius * 2) / (num_rays - 1)
                y_ray = -cone_radius + j * (cone_radius * 2) / (num_rays - 1)
                if x**2 + y_ray**2 > cone_radius**2:
                    ray_data.append((-1, np.array([0, 0, 0])))
                    continue
                ray_dir = np.array([x, y_ray, -self.DETECTION_RANGE])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                # 센서 회전 적용
                ray_dir = R @ ray_dir
                min_distance = float('inf')
                hit_vector = None
                # 바닥(y=0)과의 교차
                if abs(ray_dir[1]) > 1e-8:
                    t_plane = (0 - sensor_pos[1]) / ray_dir[1]
                    if t_plane > 0 and t_plane < min_distance:
                        hit_point = sensor_pos + t_plane * ray_dir
                        # 바닥의 x,z가 공간 내에 있는지 체크
                        if -800 <= hit_point[0] <= 800 and -800 <= hit_point[2] <= 800:
                            if t_plane <= self.DETECTION_RANGE:
                                min_distance = t_plane
                                hit_vector = hit_point - sensor_pos
                # 천장(y=800)과의 교차
                if abs(ray_dir[1]) > 1e-8:
                    t_plane = (800 - sensor_pos[1]) / ray_dir[1]
                    if t_plane > 0 and t_plane < min_distance:
                        hit_point = sensor_pos + t_plane * ray_dir
                        if -800 <= hit_point[0] <= 800 and -800 <= hit_point[2] <= 800:
                            if t_plane <= self.DETECTION_RANGE:
                                min_distance = t_plane
                                hit_vector = hit_point - sensor_pos
                # 앞/뒤 벽(z=±800)
                if abs(ray_dir[2]) > 1e-8:
                    for z_wall in [-800, 800]:
                        t_plane = (z_wall - sensor_pos[2]) / ray_dir[2]
                        if t_plane > 0 and t_plane < min_distance:
                            hit_point = sensor_pos + t_plane * ray_dir
                            if -800 <= hit_point[0] <= 800 and 0 <= hit_point[1] <= 800:
                                if t_plane <= self.DETECTION_RANGE:
                                    min_distance = t_plane
                                    hit_vector = hit_point - sensor_pos
                # 좌/우 벽(x=±800)
                if abs(ray_dir[0]) > 1e-8:
                    for x_wall in [-800, 800]:
                        t_plane = (x_wall - sensor_pos[0]) / ray_dir[0]
                        if t_plane > 0 and t_plane < min_distance:
                            hit_point = sensor_pos + t_plane * ray_dir
                            if -800 <= hit_point[2] <= 800 and 0 <= hit_point[1] <= 800:
                                if t_plane <= self.DETECTION_RANGE:
                                    min_distance = t_plane
                                    hit_vector = hit_point - sensor_pos
                # 기존 장애물 교차 판정 (구, 큐브, 박스)
                for obj in self.test_objects:
                    if obj['type'] == 'sphere':
                        sphere_center = np.array(obj['pos'])
                        sphere_radius = obj['size'] / 2
                        oc = sensor_pos - sphere_center
                        a = np.dot(ray_dir, ray_dir)
                        b = 2.0 * np.dot(oc, ray_dir)
                        c = np.dot(oc, oc) - sphere_radius * sphere_radius
                        discriminant = b * b - 4 * a * c
                        if discriminant >= 0:
                            t = (-b - math.sqrt(discriminant)) / (2.0 * a)
                            if t > 0 and t < min_distance:
                                if t <= self.DETECTION_RANGE:
                                    min_distance = t
                                    hit_point = sensor_pos + t * ray_dir
                                    hit_vector = hit_point - sensor_pos
                    elif obj['type'] == 'cube' or obj['type'] == 'rect':
                        # 박스 중심, 크기
                        box_center = np.array(obj['pos'])
                        if obj['type'] == 'cube':
                            size_x = size_y = size_z = obj['size']
                        else:
                            size_x = obj['size']
                            size_y = obj.get('size_y', obj['size'])
                            size_z = obj.get('size_z', obj['size'])
                        box_min = box_center - np.array([size_x/2, size_y/2, size_z/2])
                        box_max = box_center + np.array([size_x/2, size_y/2, size_z/2])
                        tmin = -np.inf
                        tmax = np.inf
                        for k in range(3):
                            if abs(ray_dir[k]) < 1e-8:
                                if sensor_pos[k] < box_min[k] or sensor_pos[k] > box_max[k]:
                                    tmin = np.inf
                                    tmax = -np.inf
                                    break
                            else:
                                t1 = (box_min[k] - sensor_pos[k]) / ray_dir[k]
                                t2 = (box_max[k] - sensor_pos[k]) / ray_dir[k]
                                t1, t2 = min(t1, t2), max(t1, t2)
                                tmin = max(tmin, t1)
                                tmax = min(tmax, t2)
                        if tmax >= tmin and tmin > 0 and tmin < min_distance:
                            if tmin <= self.DETECTION_RANGE:
                                min_distance = tmin
                                hit_point = sensor_pos + tmin * ray_dir
                                hit_vector = hit_point - sensor_pos
                if min_distance < float('inf'):
                    ray_data.append((min_distance, hit_vector))
                else:
                    ray_data.append((-1, np.array([0, 0, 0])))
        return ray_data

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 왼쪽 마우스 버튼
                    self.dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 3:  # 오른쪽 마우스 버튼
                    self.is_rotating = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # 휠 올림(앞으로)
                    self.space_translation[2] -= 100
                elif event.button == 5:  # 휠 내림(뒤로)
                    self.space_translation[2] += 100
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
                elif event.button == 3:
                    self.is_rotating = False
            elif event.type == pygame.MOUSEMOTION:
                if self.last_mouse_pos:
                    current_pos = pygame.mouse.get_pos()
                    dx = current_pos[0] - self.last_mouse_pos[0]
                    dy = current_pos[1] - self.last_mouse_pos[1]
                    if self.dragging:  # 왼쪽 마우스 드래그 - 공간 이동
                        self.space_translation[0] += dx * 2
                        self.space_translation[1] -= dy * 2
                    elif self.is_rotating:  # 오른쪽 마우스 드래그 - 공간 회전
                        self.space_rotation[1] += dx * 0.5
                        self.space_rotation[0] += dy * 0.5
                    self.last_mouse_pos = current_pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # R키로 회전 초기화
                    self.space_rotation = [0, 0, 0]
                elif event.key == pygame.K_LEFT:  # 왼쪽 화살표로 센서 yaw 회전 (반대방향)
                    self.sensor_rotation[1] += 5
                elif event.key == pygame.K_RIGHT:  # 오른쪽 화살표로 센서 yaw 회전 (반대방향)
                    self.sensor_rotation[1] -= 5
                elif event.key == pygame.K_UP:  # 위쪽 화살표로 센서 pitch 회전 (반대방향)
                    self.sensor_rotation[0] += 5
                elif event.key == pygame.K_DOWN:  # 아래쪽 화살표로 센서 pitch 회전 (반대방향)
                    self.sensor_rotation[0] -= 5
                elif event.key == pygame.K_y:  # Y키로 탐지범위(원뿔) 숨기기/표시 토글
                    self.show_cone = not self.show_cone

    def handle_keyboard(self):
        keys = pygame.key.get_pressed()
        move_speed = 5
        angle_rad = math.radians(self.space_rotation[1])
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        # 수직 이동 (Q/E)
        if keys[pygame.K_q]:
            self.sensor_pos[1] += 2
        if keys[pygame.K_e]:
            self.sensor_pos[1] -= 2

        # 전후 이동 (W/S)
        if keys[pygame.K_w]:
            self.sensor_pos[0] += -sin_angle * move_speed
            self.sensor_pos[2] += -cos_angle * move_speed
        if keys[pygame.K_s]:
            self.sensor_pos[0] += sin_angle * move_speed
            self.sensor_pos[2] += cos_angle * move_speed
        # 좌우 이동 (A/D)
        if keys[pygame.K_a]:
            self.sensor_pos[0] += -cos_angle * move_speed
            self.sensor_pos[2] += sin_angle * move_speed
        if keys[pygame.K_d]:
            self.sensor_pos[0] += cos_angle * move_speed
            self.sensor_pos[2] += -sin_angle * move_speed

        # 부드러운 회전 (화살표키)
        rot_speed = 2  # 더 작게 하면 더 부드럽게
        if keys[pygame.K_LEFT]:
            self.sensor_rotation[1] += rot_speed
        if keys[pygame.K_RIGHT]:
            self.sensor_rotation[1] -= rot_speed
        if keys[pygame.K_UP]:
            self.sensor_rotation[0] += rot_speed
        if keys[pygame.K_DOWN]:
            self.sensor_rotation[0] -= rot_speed

        # 바닥 충돌 처리
        if self.sensor_pos[1] < 10:  # 센서 반지름 10
            self.sensor_pos[1] = 10

    def render(self):
        # 화면 초기화
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 모델뷰 행렬 초기화
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 카메라 위치 설정 (공간 이동 반영)
        gluLookAt(0 + self.space_translation[0], 200 + self.space_translation[1], 400 + self.space_translation[2],
                  0 + self.space_translation[0], 0 + self.space_translation[1], 0 + self.space_translation[2],
                  0, 1, 0)
        
        # 공간 회전 적용
        glRotatef(self.space_rotation[0], 1, 0, 0)
        glRotatef(self.space_rotation[1], 0, 1, 0)
        glRotatef(self.space_rotation[2], 0, 0, 1)
        
        # 방(바닥, 천장, 벽) 그리기
        self.draw_room()
        
        # 그리드 그리기
        self.draw_grid()
        
        # 테스트 물체 그리기
        self.draw_test_objects()
        
        # ToF 센서 그리기
        self.draw_tof_sensor()
        
        # 화면 업데이트
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.handle_keyboard()
            self.render()
            
            # 레이캐스팅 수행
            ray_data = self.calculate_ray_hits()
            
            # 시각화 업데이트
            self.ray_visualizer.update_visualization(ray_data)
            
            pygame.time.wait(10)
        
        self.ray_visualizer.close()
        pygame.quit()

def main():
    simulator = ToFSimulator()
    simulator.run()

if __name__ == "__main__":
    main()
