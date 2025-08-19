# -*- coding: utf-8 -*-

"""3D Feature view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np
from phylib.utils import Bunch, emit
from phy.utils.color import selected_cluster_color
from phy.plot.visuals import ScatterVisual, TextVisual, LineVisual
from phy.plot.transform import range_transform, NDC
from .base import ManualClusteringView, LassoMixin

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 3D Feature view
# -----------------------------------------------------------------------------

def _get_point_color(clu_idx=None):
    """Get color for points in the 3D feature view."""
    if clu_idx is not None:
        color = selected_cluster_color(clu_idx, .5)
    else:
        color = (.5,) * 4
    assert len(color) == 4
    return color

class Feature3DView(LassoMixin, ManualClusteringView):
    """3D Feature view that shows PC components in a 3D scatter plot with rotation capability.

    Constructor
    -----------
    features : function
        Maps (cluster_id, channel_ids=None, load_all=False) to feature data.
    attributes : dict
        Maps an attribute name to a 1D array with n_spikes numbers (for example, spike times).
    channel_positions : array-like
        A (n_channels, 2) array with the x, y coordinates of the electrode sites.
    """

    # Do not show too many clusters.
    max_n_clusters = 8

    _default_position = 'right'
    cluster_ids = ()

    # Whether to disable automatic selection of channels.
    fixed_channels = False
   
    # Current axis selections
    x_axis = 'PC1 (Primary)'
    y_axis = 'PC2 (Primary)'
    z_axis = 'PC3 (Primary)'
   
    # Channel for axes (None means use primary channel)
    x_channel = None
    y_channel = None
    z_channel = None

    default_shortcuts = {
        'focus_on_cursor': 'ctrl+d',
        'add_lasso_point': 'ctrl+click',
        'clear_lasso': 'ctrl+right click',
        'pan': 'drag',
        'rotate': 'shift+drag',
        'reset_on_double_click': 'double click',
    }

    def __init__(self, features=None, attributes=None, channel_positions=None, **kwargs):
        super(Feature3DView, self).__init__(**kwargs)
        self.state_attrs += (
            'fixed_channels', 'x_axis', 'y_axis', 'z_axis', 'channel_ids', 'channel_labels',
            'projection_mode')

        assert features
        self.features = features
        self.attributes = attributes or {}
        self.channel_positions = channel_positions
        self.cluster_ids = kwargs.get('cluster_ids', [])

        # Current channels being shown
        self.channel_ids = []
        self.channel_labels = {}
       
        # Nearby channels cache
        self._nearby_channels_cache = {}
       
        # Track created action names to properly clean them up
        self._created_action_names = []
       
        # 3D transformation parameters
        self.rotation_x = 0.0  # Rotation around X axis (pitch)
        self.rotation_y = 0.0  # Rotation around Y axis (yaw) 
        self.rotation_z = 0.0  # Rotation around Z axis (roll)
        self.scale_3d = 2.5    # 3D scale factor
        self.offset_3d = np.array([0.0, 0.0])  # 2D offset after projection
        
        # Mouse interaction state
        self._mouse_pressed = False
        self._last_mouse_pos = None
        self._is_rotating = False
        self._cursor_pos = None
        
        # Data storage for 3D operations
        self._all_points_3d = None  # Original 3D coordinates
        self._all_points_cluster_id = None  # Cluster ID for each point
        self._cluster_data = []  # Store cluster data for re-projection
        
        # 3D view parameters
        self._camera_distance = 2.5
        self._fov = 45.0  # Field of view in degrees
        self._near_plane = 0.1
        self._far_plane = 1000.0

        # Projection mode (persisted). Default to orthographic to reduce distortion.
        self.projection_mode = kwargs.get('projection_mode', 'orthographic')
        
        # Enable lasso
        self.canvas.enable_lasso()

        # Create visuals
        self.visual = ScatterVisual()
        self.canvas.add_visual(self.visual)

        self.text_visual = TextVisual()
        self.canvas.add_visual(self.text_visual)

        self.axis_visual = LineVisual()
        self.canvas.add_visual(self.axis_visual)

        # Ensure marker size is set
        if not hasattr(self, '_marker_size'):
            self._marker_size = 3.0
            logger.debug("Setting default marker size to 3.0")

        # Record baseline zoom to scale marker size when zooming
        self._base_scale_3d = float(self.scale_3d)
        
        logger.debug(f"Feature3DView initialized with marker_size: {getattr(self, '_marker_size', 'NOT SET')}")

    def _find_nearby_electrodes(self, primary_channel_id, max_distance=100.0, max_count=8):
        """Find nearby electrodes within max_distance micrometers."""
        if self.channel_positions is None:
            return []
           
        # Use cache if available
        cache_key = (primary_channel_id, max_distance, max_count)
        if cache_key in self._nearby_channels_cache:
            return self._nearby_channels_cache[cache_key]
           
        if primary_channel_id >= len(self.channel_positions):
            return []
           
        primary_pos = self.channel_positions[primary_channel_id]
       
        # Calculate distances to all other channels
        distances = np.sqrt(np.sum((self.channel_positions - primary_pos) ** 2, axis=1))
       
        # Find channels within max_distance (excluding the primary channel itself)
        nearby_indices = np.where((distances <= max_distance) & (distances > 0))[0]
       
        # Sort by distance and limit to max_count
        nearby_distances = distances[nearby_indices]
        sorted_indices = np.argsort(nearby_distances)
        nearby_channels = nearby_indices[sorted_indices[:max_count]].tolist()
       
        # Cache the result
        self._nearby_channels_cache[cache_key] = nearby_channels
       
        return nearby_channels

    def _get_axis_options(self):
        """Get all available axis options for dropdowns."""
        options = []
       
        # Add PC options for primary channel
        primary_channel_text = ''
        if self.channel_ids and len(self.channel_ids) > 0:
            primary_channel = self.channel_ids[0]
            primary_channel_label = self.channel_labels.get(primary_channel, str(primary_channel))
            primary_channel_text = f" (ch{primary_channel_label})"
            
        options.extend([
            f'PC1 (Primary{primary_channel_text})',
            f'PC2 (Primary{primary_channel_text})',
            f'PC3 (Primary{primary_channel_text})'
        ])
       
        # Add time option if available
        if 'time' in self.attributes:
            options.append('time')
       
        # Add PC options for nearby channels if we have a primary channel
        if self.channel_ids and len(self.channel_ids) > 0:
            try:
                primary_channel = self.channel_ids[0]
                nearby_channels = self._find_nearby_electrodes(primary_channel)
               
                for i, channel_id in enumerate(nearby_channels):
                    if channel_id == primary_channel:
                        continue
                    channel_label = self.channel_labels.get(channel_id, str(channel_id))
                    for pc in ['PC1', 'PC2', 'PC3']:
                        options.append(f'{channel_label}_{pc}')
            except Exception as e:
                logger.debug(f"Could not get nearby electrodes: {e}")
       
        return options

    def _parse_axis_option(self, option):
        """Parse an axis option string into channel and PC."""
        if '(Primary' in option:
            pc = option.split(' ')[0]
            return None, pc  # Primary channel
        elif option == 'time':
            return None, 'time'
        elif option in ['PC1', 'PC2', 'PC3']:
            return None, option  # Primary channel
        elif '_' in option:
            # Format like 'ch32_PC1'
            parts = option.split('_')
            if len(parts) == 2:
                channel_label, pc = parts
                # Find channel ID from label
                for channel_id, label in self.channel_labels.items():
                    if label == channel_label:
                        return channel_id, pc
        return None, 'PC1'  # Default fallback

    def _get_axis_data(self, bunch, axis_option, cluster_id=None):
        """Extract data for a specific axis."""
        channel_id, pc = self._parse_axis_option(axis_option)
       
        if pc == 'time':
            if 'time' in self.attributes:
                time_bunch = self.attributes['time'](cluster_id, load_all=False)
                result = time_bunch.data if hasattr(time_bunch, 'data') else time_bunch
                return result
            else:
                spike_ids = bunch.get('spike_ids', [])
                if spike_ids is None:
                    spike_ids = []
                return np.zeros(len(spike_ids))
       
        # For PC data
        if channel_id is None:
            channel_ids = bunch.get('channel_ids', [])
            if len(channel_ids) == 0:
                return np.zeros(bunch.get('data', np.array([])).shape[0])
            channel_id = channel_ids[0]
       
        # Get the column index of the channel in the data
        channel_ids = bunch.get('channel_ids', [])
        if channel_id not in channel_ids:
            return np.zeros(bunch.get('data', np.array([])).shape[0])
       
        channel_idx = list(channel_ids).index(channel_id)
        pc_idx = {'PC1': 0, 'PC2': 1, 'PC3': 2}.get(pc, 0)
       
        data = bunch.get('data', np.array([]))
       
        if data.ndim == 3 and data.shape[2] > pc_idx:
            return data[:, channel_idx, pc_idx]
        else:
            return np.zeros(data.shape[0] if data.ndim > 0 else 0)

    def _create_rotation_matrix(self):
        """Create a 3D rotation matrix from current rotation angles."""
        # Rotation matrices for each axis
        cos_x, sin_x = np.cos(self.rotation_x), np.sin(self.rotation_x)
        cos_y, sin_y = np.cos(self.rotation_y), np.sin(self.rotation_y)
        
        # Individual rotation matrices
        R_x = np.array([[1, 0, 0],
                       [0, cos_x, -sin_x],
                       [0, sin_x, cos_x]])
        
        R_y = np.array([[cos_y, 0, sin_y],
                       [0, 1, 0],
                       [-sin_y, 0, cos_y]])
        
        # Combined rotation: apply Yaw, then Pitch for intuitive orbiting
        return R_x @ R_y

    def _project_3d_to_2d(self, points_3d, return_depth=False):
        """Project 3D points to 2D using perspective projection.
        
        If return_depth is True, also returns a depth measure per point for sizing/sorting.
        """
        if len(points_3d) == 0:
            return (points_3d, np.array([]), np.array([])) if return_depth else points_3d
        
        # Center rotation around the mean of the selected clusters if available
        def _get_rotation_center():
            if self._all_points_3d is None or len(self._all_points_3d) == 0:
                return np.array([0.0, 0.0, 0.0])
            try:
                if getattr(self, '_all_points_cluster_id', None) is not None and self.cluster_ids:
                    mask = np.isin(self._all_points_cluster_id, self.cluster_ids)
                    if np.any(mask):
                        return np.mean(self._all_points_3d[mask], axis=0)
            except Exception:
                pass
            return np.mean(self._all_points_3d, axis=0)
        
        center = _get_rotation_center()
        centered_points = points_3d - center
        
        # Apply rotation
        rotation_matrix = self._create_rotation_matrix()
        rotated_points = centered_points @ rotation_matrix.T
        
        # Apply 3D scaling
        scaled_points = rotated_points * self.scale_3d
        
        if getattr(self, 'projection_mode', 'orthographic') == 'orthographic':
            # Orthographic projection: no division by depth
            x_proj = scaled_points[:, 0]
            y_proj = scaled_points[:, 1]
            # Use raw Z (shifted) for depth ordering only
            z_camera = scaled_points[:, 2] + self._camera_distance
            # Constant size factor in ortho
            size_factor = np.ones_like(z_camera)
        else:
            # Perspective projection
            # Move camera back along Z axis
            z_camera = scaled_points[:, 2] + self._camera_distance
            # Prevent division by zero or negative values
            z_camera = np.maximum(z_camera, 0.1)
            # Convert field of view to focal length
            focal_length = 1.0 / np.tan(np.radians(self._fov / 2.0))
            # Perspective projection
            x_proj = (scaled_points[:, 0] * focal_length) / z_camera
            y_proj = (scaled_points[:, 1] * focal_length) / z_camera
            # Depth-based size factor (may be overridden by caller)
            size_factor = 1.0 + 0.6 * (1.0 / z_camera)
        
        # Combine to 2D points
        points_2d = np.column_stack([x_proj, y_proj])
        
        # Apply 2D offset
        points_2d += self.offset_3d
        
        if return_depth:
            return points_2d, size_factor, z_camera
        
        return points_2d

    def _get_data_bounds(self, bunchs):
        """Compute data bounds for the view."""
        if not bunchs:
            return (-1, -1, 1, 1)

        xs, ys, zs = [], [], []
        for bunch in bunchs:
            x = self._get_axis_data(bunch, self.x_axis, bunch.cluster_id)
            y = self._get_axis_data(bunch, self.y_axis, bunch.cluster_id)
            z = self._get_axis_data(bunch, self.z_axis, bunch.cluster_id)
            if x is None or y is None or z is None:
                continue
            if len(x) == 0 or len(y) == 0 or len(z) == 0:
                continue
            xs.append(x)
            ys.append(y)
            zs.append(z)

        if not xs:
            return (-1, -1, 1, 1)

        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        z_all = np.concatenate(zs)

        # Fallback if non-finite or degenerate
        if not np.all(np.isfinite([x_all.min(), x_all.max(), y_all.min(), y_all.max(), z_all.min(), z_all.max()])):
            return (-1, -1, 1, 1)

        # Avoid zero ranges by padding slightly
        def _pad(vmin, vmax):
            if vmax <= vmin:
                return vmin - 1.0, vmax + 1.0
            span = vmax - vmin
            pad = 0.05 * span if span > 0 else 1.0
            return vmin - pad, vmax + pad

        x_min, x_max = _pad(x_all.min(), x_all.max())
        y_min, y_max = _pad(y_all.min(), y_all.max())
        z_min, z_max = _pad(z_all.min(), z_all.max())

        # Store 3D bounds for potential use elsewhere
        self._data_bounds_3d = [x_min, y_min, z_min, x_max, y_max, z_max]

        # Return 2D bounds for canvas normalization
        return (x_min, y_min, x_max, y_max)

    def get_clusters_data(self, fixed_channels=None, load_all=None):
        """Get feature data for all selected clusters."""
        logger.debug(f"get_clusters_data called with cluster_ids: {self.cluster_ids}")

        # Get the feature data.
        c = self.channel_ids if fixed_channels else None
        logger.debug(f"Calling features() with channel_ids: {c}")

        bunchs = self.features(self.cluster_ids, channel_ids=c, load_all=load_all)
        if bunchs is None:
            bunchs = []

        logger.debug(f"Got {len(bunchs)} raw bunches")

        if not bunchs:
            logger.debug("No valid bunches, returning empty list")
            return []

        # Choose the channels based on the first selected cluster.
        channel_ids = list(bunchs[0].get('channel_ids', [])) if bunchs else []
        logger.debug(f"Extracted channel_ids from first bunch: {channel_ids[:5] if len(channel_ids) > 5 else channel_ids}")
       
        # Always update channel_ids if not in fixed_channels mode
        if not fixed_channels:
            logger.debug(f"Setting channel_ids to {channel_ids[:5]}..." if len(channel_ids) > 5 else f"Setting channel_ids to {channel_ids}")
            self.channel_ids = channel_ids
       
        # Channel labels: get the proper electrode numbers, not just indices
        self.channel_labels = {}
        for d in bunchs:
            channel_ids_bunch = d.get('channel_ids', [])
            channel_labels = d.get('channel_labels', [])
           
            # If no channel labels provided, use the channel IDs themselves
            if not channel_labels:
                channel_labels = [str(ch) for ch in channel_ids_bunch]
           
            # Map channel_id to its proper label
            for i, channel_id in enumerate(channel_ids_bunch):
                if i < len(channel_labels):
                    self.channel_labels[channel_id] = channel_labels[i]
                else:
                    self.channel_labels[channel_id] = str(channel_id)

        logger.debug(f"Final channel_labels: {dict(list(self.channel_labels.items())[:3])}")
        return bunchs

    def _update_projections(self):
        """Update 2D projections of all data without full replot."""
        if not self._cluster_data:
            return
       
        # Clear visuals
        self.visual.reset_batch()
        self.text_visual.reset_batch()
        self.axis_visual.reset_batch()
        
        # --- Start of Global Sorting Implementation ---
        
        # 1. Collect all data from all clusters
        all_points_3d = []
        all_colors = []
        all_spike_ids = []
        all_bunches = []
        
        # Add a flag to track if normalization has been applied
        if not hasattr(self, '_is_normalized'):
            self._is_normalized = False

        # Ensure 2D offset starts at zero for a new projection cycle
        if not hasattr(self, 'offset_3d'):
            self.offset_3d = np.array([0.0, 0.0])

        for cluster_info in self._cluster_data:
            points_3d = cluster_info['points_3d']
            if len(points_3d) == 0:
                continue
            
            all_points_3d.append(points_3d)
            # Repeat the color for each point in the cluster
            all_colors.extend([cluster_info['color']] * len(points_3d))
            all_spike_ids.extend(cluster_info['spike_ids'])
            all_bunches.append(cluster_info['bunch'])
            
        if not all_points_3d:
            self.canvas.update()
            return
            
        # 2. Project all points at once to get depth information
        global_points_3d = np.vstack(all_points_3d)
        global_points_2d, size_factor, z_cam = self._project_3d_to_2d(global_points_3d, return_depth=True)
 
        # 3. Create a global sort order (farthest to nearest)
        global_order = np.argsort(z_cam)[::-1]
        
        # 4. Apply the sort order to all data attributes
        sorted_points_2d = global_points_2d[global_order]
        sorted_colors = np.array(all_colors)[global_order]
        # Keep marker size constant during rotation to avoid perceived zoom
        sorted_size_factor = np.ones_like(size_factor[global_order])
 
        # 5. Add to visual with depth-based sizing and correct order
        base_size = getattr(self, '_marker_size', 3.0)
        # Increase point size with zoom (uniformly, no depth scaling)
        zoom_factor = 1.0
        try:
            zoom_factor = float(self.scale_3d) / float(getattr(self, '_base_scale_3d', self.scale_3d))
            # Clamp to reasonable range
            zoom_factor = float(np.clip(zoom_factor, 0.5, 3.0))
        except Exception:
            zoom_factor = 1.0

        # Set fixed data bounds since points are projected to [-1,1]
        self.data_bounds = (-1, -1, 1, 1)

        self.visual.add_batch_data(
            pos=sorted_points_2d,
            color=sorted_colors,
            size=base_size * zoom_factor * sorted_size_factor,
            data_bounds=self.data_bounds
        )

        # --- End of Global Sorting Implementation ---

        # Update lasso data with unsorted points
        start_idx = 0
        for i, bunch in enumerate(all_bunches):
            num_points = len(self._cluster_data[i]['points_3d'])
            end_idx = start_idx + num_points
            bunch.pos = global_points_2d[start_idx:end_idx]
            start_idx = end_idx
       
        # Update axes
        self._plot_axes()
        
        # Update canvas
        self.canvas.update_visual(self.visual)
        self.canvas.update_visual(self.text_visual)
        self.canvas.update_visual(self.axis_visual)
        self.visual.show()
        self.canvas.update()

    def on_request_split(self, sender=None):
        """Return the spikes enclosed by the lasso."""
        if not hasattr(self.canvas, 'lasso') or not self.canvas.lasso:
            logger.debug("No lasso selection available")
            return np.array([], dtype=np.int64)

        # Get lasso polygon
        lasso_points = self.canvas.lasso.polygon
        if len(lasso_points) < 3:
            logger.debug("Lasso polygon too small")
            return np.array([], dtype=np.int64)

        # Find points inside the lasso for each cluster
        spike_ids_to_split = []

        for cluster_info in self._cluster_data:
            if cluster_info['cluster_id'] is None:  # Skip background
                continue

            bunch = cluster_info['bunch']
            if hasattr(bunch, 'pos') and len(bunch.pos) > 0:
                pts2d = bunch.pos
            else:
                # Fallback: compute 2D positions from stored 3D points
                pts2d = self._project_3d_to_2d(cluster_info['points_3d'])
                bunch.pos = pts2d
            # Check which points are inside the lasso
            from matplotlib.path import Path
            lasso_path = Path(lasso_points)
            inside_mask = lasso_path.contains_points(pts2d)

            if np.any(inside_mask):
                spike_ids = bunch.get('spike_ids', [])
                if spike_ids is not None and len(spike_ids) > 0:
                    selected_spikes = np.array(spike_ids)[inside_mask]
                    spike_ids_to_split.extend(selected_spikes.tolist())

        if spike_ids_to_split:
            logger.info(f"Splitting {len(spike_ids_to_split)} spikes with lasso")
            # Clear the lasso after splitting
            self.canvas.lasso.clear()
        else:
            logger.debug("No spikes selected by lasso")

        return np.array(spike_ids_to_split, dtype=np.int64)

    def plot(self, **kwargs):
        """Update the view with the selected clusters."""
        logger.debug("Feature3DView.plot() called")
        if not self.cluster_ids:
            logger.debug("No clusters selected, clearing view.")
            self.visual.reset_batch()
            self.text_visual.reset_batch()
            self.axis_visual.reset_batch()
            self.visual.hide()
            self.canvas.update()
            self._cluster_data = []
            return

        if hasattr(self, 'canvas') and hasattr(self.canvas, 'lasso'):
            self.canvas.lasso.clear()

        try:
            # Get the clusters data.
            bunchs = self.get_clusters_data(fixed_channels=self.fixed_channels)
            logger.debug(f"Got {len(bunchs)} cluster bunches")

            # Update axis selections if they are set to primary
            if self.channel_ids and len(self.channel_ids) > 0:
                primary_channel = self.channel_ids[0]
                primary_channel_label = self.channel_labels.get(primary_channel, str(primary_channel))
                primary_channel_text = f" (ch{primary_channel_label})"

                # Preserve the chosen PC (PC1/PC2/PC3) for each axis when relabeling Primary
                for axis_name in ('x', 'y', 'z'):
                    current_label = getattr(self, f'{axis_name}_axis')
                    if '(Primary' in current_label:
                        pc = current_label.split(' ')[0]  # e.g., 'PC1'
                        setattr(self, f'{axis_name}_axis', f"{pc} (Primary{primary_channel_text})")

            if not bunchs:
                logger.debug("No cluster data, clearing view")
                self.visual.reset_batch()
                self.text_visual.reset_batch()
                self.axis_visual.reset_batch()
                self.visual.hide()
                self.canvas.update()
                self._cluster_data = []
                return
       
            # Calculate data bounds
            self.data_bounds = self._get_data_bounds(bunchs)
       
            # When replotting, clear normalization so axes recompute sizes
            self._is_normalized = False

            # Clear previous data
            self._cluster_data = []
            self.visual.reset_batch()
            self.text_visual.reset_batch()
            self.axis_visual.reset_batch()

            all_points_3d = []
            all_cluster_ids = []
       
            # Get and plot background data (gray points)
            if self.channel_ids:
                logger.debug("Getting background data")
                background_data = self.features(None, channel_ids=self.channel_ids)
                # Handle both list and single-bunch returns
                background = background_data[0] if isinstance(background_data, (list, tuple)) and background_data else background_data
                if background:
                    background.cluster_id = None
                    x_bg = self._get_axis_data(background, self.x_axis)
                    y_bg = self._get_axis_data(background, self.y_axis)
                    z_bg = self._get_axis_data(background, self.z_axis)
                    points_3d = np.column_stack([x_bg, y_bg, z_bg])
                   
                    # Store cluster data
                    cluster_info = {
                        'points_3d': points_3d,
                        'cluster_id': None,
                        'clu_idx': None,
                        'color': _get_point_color(None),
                        'spike_ids': background.get('spike_ids'),
                        'bunch': background
                    }
                    self._cluster_data.append(cluster_info)
                    
                    all_points_3d.append(points_3d)
                    all_cluster_ids.extend([None] * len(points_3d))
       
            # Plot each cluster
            for clu_idx, bunch in enumerate(bunchs):
                cluster_id = bunch.cluster_id
                x = self._get_axis_data(bunch, self.x_axis, cluster_id)
                y = self._get_axis_data(bunch, self.y_axis, cluster_id)
                z = self._get_axis_data(bunch, self.z_axis, cluster_id)
                points_3d = np.column_stack([x, y, z])
               
                # Store cluster data
                cluster_info = {
                    'points_3d': points_3d,
                    'cluster_id': cluster_id,
                    'clu_idx': clu_idx,
                    'color': _get_point_color(clu_idx),
                    'spike_ids': bunch.get('spike_ids'),
                    'bunch': bunch
                }
                self._cluster_data.append(cluster_info)
                
                all_points_3d.append(points_3d)
                all_cluster_ids.extend([cluster_id] * len(points_3d))

            # Store all points for global operations
            if all_points_3d:
                self._all_points_3d = np.vstack(all_points_3d)
                self._all_points_cluster_id = np.array(all_cluster_ids)
                self._is_normalized = False  # Mark as not normalized
            else:
                self._all_points_3d = None
                self._all_points_cluster_id = None

            # Recenter and rescale the view to fit the new data.
            self._autoscale_and_recenter()

            # Update projections and render
            self._update_projections()
            # Make sure the canvas is not left in lazy/panned state
            if hasattr(self.canvas, 'panzoom'):
                self.canvas.panzoom.enabled = True

            logger.debug("Feature3DView.plot() completed")

        except Exception as e:
            logger.error(f"Error in Feature3DView.plot(): {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _plot_axes(self):
        """Add 3D axis lines and labels."""
        # Create 3D axis endpoints
        axis_length = 5.0
        axes_3d = np.array([
            [0, 0, 0],  # Origin
            [axis_length, 0, 0],  # X axis end
            [0, axis_length, 0],  # Y axis end
            [0, 0, axis_length],  # Z axis end
        ])
       
        # Project to 2D
        axes_2d = self._project_3d_to_2d(axes_3d)
        origin_2d = axes_2d[0]
        x_end, y_end, z_end = axes_2d[1], axes_2d[2], axes_2d[3]
       
        # Colors for axes
        colors = [
            (1.0, 0.0, 0.0, 0.8),  # Red for X
            (0.0, 1.0, 0.0, 0.8),  # Green for Y  
            (0.0, 0.0, 1.0, 0.8),  # Blue for Z
        ]
       
        labels = [self.x_axis, self.y_axis, self.z_axis]
        endpoints = [x_end, y_end, z_end]
        
        for end_pos, label, color in zip(endpoints, labels, colors):
            # Add text label
            self.text_visual.add_batch_data(
                pos=end_pos,
                text=label,
                color=color,
                anchor=(0, 0),
                data_bounds=self.data_bounds
            )
            
            # Add axis line - LineVisual expects (n_lines, 4) format [x1, y1, x2, y2]
            line_pos = np.array([origin_2d[0], origin_2d[1], end_pos[0], end_pos[1]]).reshape((1, 4))
            self.axis_visual.add_batch_data(
                pos=line_pos,
                color=color,
                data_bounds=self.data_bounds,
            )

    def on_select(self, cluster_ids=(), **kwargs):
        """Called when clusters are selected."""
        logger.debug(f"on_select called with cluster_ids: {cluster_ids}")
       
        # Call parent class method first
        super(Feature3DView, self).on_select(cluster_ids=cluster_ids, **kwargs)

        # Auto-center/focus on new selection: reset PanZoom and 2D offset
        if hasattr(self.canvas, 'panzoom'):
            try:
                self.canvas.panzoom.reset()
            except Exception:
                pass
        self.offset_3d = np.array([0.0, 0.0])

        # Update axis actions when clusters change
        self._create_axis_actions()
        # Update status bar
        self.update_status()

    def on_mouse_press(self, e):
        """Handle mouse press to initiate rotation."""
        if e.button == 'Right':
            return

        self._mouse_pressed = True
        self._last_mouse_pos = e.pos

        # Check for rotation (Shift + drag)
        if 'Shift' in e.modifiers:
            self._is_rotating = True
            # Suppress PanZoom during rotation so it doesn't pan the 2D view
            if hasattr(self.canvas, 'panzoom'):
                pz = self.canvas.panzoom
                self._saved_panzoom_enabled = getattr(pz, 'enabled', True)
                self._saved_mouse_pressed = getattr(pz, '_mouse_pressed', False)
                pz.enabled = False
                pz._mouse_pressed = False
        else:
            self._is_rotating = False

    def on_mouse_move(self, e):
        """Handle mouse move for rotation."""
        self._cursor_pos = e.pos
       
        if self._mouse_pressed and self._is_rotating:
            if not self._last_mouse_pos:
                self._last_mouse_pos = e.pos
                return
                
            dx = e.pos[0] - self._last_mouse_pos[0]
            dy = e.pos[1] - self._last_mouse_pos[1]
            
            # Skip tiny movements
            if abs(dx) < 1 and abs(dy) < 1:
                return

            # Update rotation angles
            rotation_sensitivity = 0.01
            self.rotation_y += dx * rotation_sensitivity  # Yaw (left/right)
            self.rotation_x += dy * rotation_sensitivity  # Pitch (up/down)
 
            # Clamp pitch
            self.rotation_x = float(np.clip(self.rotation_x, -1.3, 1.3))
            
            # Update projections
            self._update_projections()
            self._last_mouse_pos = e.pos

    def on_key_press(self, e):
        """Keyboard rotation controls."""
        key = getattr(e, 'key', None)
        if not key:
            return
            
        step = 0.05
        updated = False
        
        if key in ('A', 'a'):
            self.rotation_y -= step
            updated = True
        elif key in ('D', 'd'):
            self.rotation_y += step
            updated = True
        elif key in ('W', 'w'):
            self.rotation_x -= step
            updated = True
        elif key in ('S', 's'):
            self.rotation_x += step
            updated = True
            
        # Clamp pitch
        self.rotation_x = float(np.clip(self.rotation_x, -1.3, 1.3))
        
        if updated:
            self._update_projections()

    def on_mouse_release(self, e):
        """Handle mouse release."""
        self._mouse_pressed = False
        self._is_rotating = False
        self._last_mouse_pos = None
        # Restore PanZoom handling if we disabled it
        if hasattr(self.canvas, 'panzoom') and hasattr(self, '_saved_panzoom_enabled'):
            pz = self.canvas.panzoom
            pz.enabled = getattr(self, '_saved_panzoom_enabled', True)
            pz._mouse_pressed = getattr(self, '_saved_mouse_pressed', False)
            self._saved_panzoom_enabled = None
            self._saved_mouse_pressed = None

    def on_mouse_double_click(self, e):
        """Reset the view on double click."""
        self._cursor_pos = e.pos
        logger.info("Double-click registered to reset view.")
        self.reset_view()

    def on_mouse_wheel(self, e):
        """Handle zoom with mouse wheel."""
        # Adjust 3D scale
        zoom_factor = 1.1 if e.delta > 0 else 1.0 / 1.1
        self.scale_3d *= zoom_factor
        self._update_projections()

    def focus_on_cursor(self):
        """Focus the view on the cursor position."""
        if self._cursor_pos is None:
            logger.debug("No cursor position available")
            return
        logger.info("Focus on cursor called")

    def reset_view(self):
        """Reset the 3D view to default state."""
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self._is_normalized = False
        
        # Also reset phy's PanZoom
        if hasattr(self.canvas, 'panzoom'):
            self.canvas.panzoom.reset()
        
        # Use auto-scaling to set appropriate zoom level
        self._autoscale_and_recenter()
        self._update_projections()

    def _autoscale_and_recenter(self):
        """Automatically adjust scale and reset offset to frame the data.

        This method calculates the bounding box of the currently displayed 3D points
        and adjusts the `scale_3d` and `offset_3d` to ensure the data is
        centered and fully visible, without changing the current camera rotation.
        """
        if self._all_points_3d is None or len(self._all_points_3d) == 0:
            return

        # Use normalized points if they have been computed.
        points_to_frame = self._all_points_3d
        
        # Always calculate the actual data bounds for proper scaling
        min_vals = np.min(points_to_frame, axis=0)
        max_vals = np.max(points_to_frame, axis=0)

        # Get the maximum span in any single dimension (not diagonal)
        # This ensures the data fills the view nicely
        data_span = np.max(max_vals - min_vals)
        if data_span < 1e-6:
            data_span = 1.0  # Avoid division by zero for single points.

        # Scale so the largest dimension fills 100% of the view.
        # Since view goes from -1 to 1 (span of 2), we want data_span * scale = 2.0
        new_scale = 2.0 / data_span
        self.scale_3d = float(new_scale)
        self._base_scale_3d = float(new_scale)

        # Reset any 2D panning.
        self.offset_3d = np.array([0.0, 0.0])

        # Also reset phy's native PanZoom to ensure a clean state.
        if hasattr(self.canvas, 'panzoom'):
            self.canvas.panzoom.reset()

        logger.debug(f"Autoscaled to new scale: {self.scale_3d}")

    def zoom_in(self):
        """Zoom in the 3D view."""
        self.scale_3d *= 1.1
        self._update_projections()

    def zoom_out(self):
        """Zoom out the 3D view."""
        self.scale_3d /= 1.1
        self._update_projections()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(Feature3DView, self).attach(gui)

        # Add actions - the shortcuts are automatically handled by the Actions system
        self.actions.add(self.zoom_in, name='Zoom in')
        self.actions.add(self.zoom_out, name='Zoom out')
        self.actions.add(self.reset_view, name='Reset view')
        self.actions.separator()
        # Register the toggle action so the default shortcut is picked up.
        self.actions.add(
            self.toggle_automatic_channel_selection,
            checkable=True,
            checked=not self.fixed_channels,
        )

        # Projection toggle (Orthographic/Perspective)
        self.actions.add(
            self.toggle_projection_mode,
            name='Orthographic projection',
            checkable=True,
            checked=(self.projection_mode == 'orthographic')
        )

        # Create axis actions at startup
        self._create_axis_actions()

        # Force an initial plot to ensure the view is not blank on startup.
        self.plot()

    def toggle_projection_mode(self, checked):
        """Toggle between orthographic and perspective projection."""
        self.projection_mode = 'orthographic' if checked else 'perspective'
        self._is_normalized = False
        self._update_projections()

    def _create_axis_actions(self):
        """Create or update axis selection actions."""
        # Clear old actions
        if hasattr(self.actions, '_view_submenus'):
            for submenu_name, qmenu in self.actions._view_submenus.items():
                if submenu_name.startswith('Set '):
                    qmenu.clear()

        # Remove old actions
        for action_name in self._created_action_names:
            if action_name in self.actions._actions_dict:
                try:
                    self.actions.remove(action_name)
                except Exception:
                    pass
        self._created_action_names.clear()

        # Get dynamic axis options
        axis_options = self._get_axis_options()

        # Build default labels for primary channel for fallback
        primary_channel_text = ''
        if self.channel_ids and len(self.channel_ids) > 0:
            primary_channel = self.channel_ids[0]
            primary_label = self.channel_labels.get(primary_channel, str(primary_channel))
            primary_channel_text = f" (ch{primary_label})"
        default_axis_label = {
            'x': f'PC1 (Primary{primary_channel_text})',
            'y': f'PC2 (Primary{primary_channel_text})',
            'z': f'PC3 (Primary{primary_channel_text})',
        }

        # Action callbacks
        def _make_axis_action(axis_name, option):
            def callback(checked=False):
                setattr(self, f'{axis_name}_axis', option)
                # Force a full plot and menu refresh to ensure state is clean
                self.plot()
                self._create_axis_actions()
                self.update_status()
            return callback

        def _make_reset_action(axis_name):
            def callback():
                default_value = {
                    'x': 'PC1 (Primary)',
                    'y': 'PC2 (Primary)',
                    'z': 'PC3 (Primary)',
                }[axis_name]
                setattr(self, f'{axis_name}_axis', default_value)
                self.plot()
                self._create_axis_actions()  # Recreate to update checkmarks
                self.update_status()
            return callback

        for axis_name in ['x', 'y', 'z']:
            submenu_name = f'Set {axis_name.upper()} axis to'
            current_axis_value = getattr(self, f'{axis_name}_axis')

            # Fallback if current selection is not available in options
            if current_axis_value not in axis_options:
                fallback = default_axis_label[axis_name]
                setattr(self, f'{axis_name}_axis', fallback)
                current_axis_value = fallback

            # Add Reset action first
            reset_name = f'Reset {axis_name.upper()} axis to default'
            self.actions.add(
                _make_reset_action(axis_name),
                name=reset_name,
                view_submenu=submenu_name,
            )
            self._created_action_names.append(reset_name)
            self.actions.separator(view_submenu=submenu_name)

            for option in axis_options:
                action_name = f'Set {axis_name.upper()} to {option}'
                self._created_action_names.append(action_name)

                self.actions.add(
                    _make_axis_action(axis_name, option),
                    name=action_name,
                    view_submenu=submenu_name,
                    checkable=True,
                    checked=(option == current_axis_value),
                    show_shortcut=False
                )
        self.canvas.update()

    def toggle_automatic_channel_selection(self, checked):
        """Toggle the automatic selection of channels when the cluster selection changes."""
        self.fixed_channels = not checked
        # Update the checkbox in the menu.
        action = self.actions.get('toggle_automatic_channel_selection')
        if action:
            action.setChecked(not self.fixed_channels)

        # If re-enabling automatic selection, reset to primary defaults and replot.
        if not self.fixed_channels:
            # Get primary channel info for the labels
            primary_channel_text = ''
            if self.channel_ids and len(self.channel_ids) > 0:
                primary_channel = self.channel_ids[0]
                primary_channel_label = self.channel_labels.get(primary_channel, str(primary_channel))
                primary_channel_text = f" (ch{primary_channel_label})"

            self.x_axis = f'PC1 (Primary{primary_channel_text})'
            self.y_axis = f'PC2 (Primary{primary_channel_text})'
            self.z_axis = f'PC3 (Primary{primary_channel_text})'

            # Re-plot to apply the change immediately.
            self.plot()
            # Update the menu checkboxes to reflect the current axis values
            self._create_axis_actions()

        self.update_status()
 
    def update_status(self):
        """Update the status bar."""
        emit('status', self, self.status)

    @property
    def status(self):
        """Status bar text."""
        if not self.channel_ids or len(self.channel_ids) == 0:
            return 'No channels selected'
    
        # Get the proper channel label
        primary_channel_id = self.channel_ids[0]
        primary_channel_label = self.channel_labels.get(primary_channel_id, str(primary_channel_id))
        
        channel_mode = 'Fixed' if self.fixed_channels else 'Auto'
        
        return (f'3D Feature View - X:{self.x_axis}, Y:{self.y_axis}, Z:{self.z_axis} | '
                f'Primary: ch{primary_channel_label} ({channel_mode}) | '
                f'Rotation: X={self.rotation_x:.2f}, Y={self.rotation_y:.2f}, Z={self.rotation_z:.2f} | '
                f'Scale: {self.scale_3d:.2f}')
                