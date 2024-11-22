class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        num_frames: int = 32,
        hidden_dim: int = 768,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Backbone
        backbone = models.efficientnet_b4(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        
        # Multi-scale feature extraction
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1792, hidden_dim, kernel_size=3, padding=2**i, dilation=2**i),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ) for i in range(3)
        ])
        
        # Feature processing modules
        self.freq_attention = FrequencyAttention(hidden_dim * 3)
        self.temporal_module = TemporalConsistencyModule(hidden_dim * 3)
        self.spatial_attention = SpatialAttention(hidden_dim * 3)
        
        # Frame reconstruction for self-supervision
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 3, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> dict:
        B, C, T, H, W = x.shape
        
        # Process each frame
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        features = self.encoder(x)
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for scale_module in self.multi_scale:
            scaled_features = scale_module(features)
            multi_scale_features.append(scaled_features)
        
        features = torch.cat(multi_scale_features, dim=1)
        
        # Apply attention mechanisms
        features = self.freq_attention(features)
        features = rearrange(features, '(b t) c h w -> b c t h w', b=B)
        features = self.temporal_module(features)
        features = rearrange(features, 'b c t h w -> (b t) c h w')
        features = self.spatial_attention(features)
        
        # Classification
        logits = self.classifier(features)
        logits = rearrange(logits, '(b t) d -> b t d', b=B)
        logits = logits.mean(dim=1)  # Average over frames
        
        # Reconstruction for self-supervision
        reconstruction = self.decoder(features)
        
        return {
            'logits': logits,
            'features': features,
            'reconstruction': reconstruction
        }