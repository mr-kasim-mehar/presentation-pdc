# ReactBits Components List

All available ReactBits components that can be imported directly from `@appletosolutions/reactbits`:

## Text Components
- `SplitText` - Split text animation
- `BlurText` - Blur text effect
- `CircularText` - Circular text layout
- `ShinyText` - Shiny text effect
- `TextPressure` - Text pressure effect
- `FuzzyText` - Fuzzy text effect
- `GradientText` - Gradient text
- `FallingText` - Falling text animation
- `TextCursor` - Text cursor effect
- `DecryptedText` - Decrypted text effect
- `AsciiText` - ASCII text
- `ScrambleText` - Scrambled text
- `GlitchText` - Glitch text effect
- `LetterGlitch` - Letter glitch effect

## Animation Components
- `Bounce` - Bounce animation
- `ScrollFloat` - Scroll float effect
- `ScrollReveal` - Scroll reveal animation
- `ScrollVelocity` - Scroll velocity effect
- `AnimatedContent` - Animated content
- `FadeContent` - Fade content
- `AnimatedList` - Animated list

## Interactive Components
- `ClickSpark` - Click spark effect
- `TrueFocus` - True focus effect
- `VariableProximity` - Variable proximity
- `Magnet` - Magnet effect
- `MagnetLines` - Magnet lines
- `BlobCursor` - Blob cursor
- `SplashCursor` - Splash cursor
- `Crosshair` - Crosshair
- `PixelTrail` - Pixel trail

## Background Components
- `Silk` - Silk background (currently using)
- `Aurora` - Aurora background
- `Beams` - Beams background
- `Lightning` - Lightning effect
- `Noise` - Noise background
- `Dither` - Dither effect
- `Balatro` - Balatro background
- `ShapeBlur` - Shape blur
- `DotGrid` - Dot grid (we used this before)
- `Threads` - Threads background
- `Hyperspeed` - Hyperspeed background
- `Iridescence` - Iridescence background
- `GridDistortion` - Grid distortion
- `Ballpit` - Ballpit background
- `Orb` - Orb background
- `GridMotion` - Grid motion
- `LiquidChrome` - Liquid chrome
- `Squares` - Squares background
- `Waves` - Waves background
- `MetallicPaint` - Metallic paint
- `Ribbons` - Ribbons background

## Card Components
- `Stack` - Stack of cards
- `TiltedCard` - Tilted card
- `ChromaGrid` - Chroma grid
- `ProfileCard` - Profile card
- `PixelCard` - Pixel card
- `SpotlightCard` - Spotlight card
- `DecayCard` - Decay card
- `CardSwap` - Card swap
- `BounceCards` - Bounce cards

## Gallery/Media Components
- `CircularGallery` - Circular gallery
- `Carousel` - Carousel
- `FlyingPosters` - Flying posters
- `RollingGallery` - Rolling gallery
- `ImageTrail` - Image trail
- `ModelViewer` - 3D model viewer

## Navigation Components
- `Dock` - Dock navigation
- `GooeyNav` - Gooey navigation
- `FlowingMenu` - Flowing menu
- `InfiniteMenu` - Infinite menu

## Other Components
- `Folder` - Folder component
- `GlassIcons` - Glass icons
- `InfiniteScroll` - Infinite scroll
- `ElasticSlider` - Elastic slider
- `Counter` - Counter
- `CountUp` - Count up animation
- `Stepper` - Stepper component
- `GlareHover` - Glare hover effect
- `PixelTransition` - Pixel transition
- `Particles` - Particles effect

## Usage Examples

### Direct Import (Recommended)
```tsx
import { Silk, Aurora, Beams, DotGrid } from '@appletosolutions/reactbits';

// Use in your component
<Silk speed={5} scale={1} color="#5227FF" />
<Aurora />
<Beams />
<DotGrid dotSize={16} gap={32} baseColor="#5227FF" />
```

### Via shadcn CLI
```bash
npx shadcn@latest add @react-bits/ComponentName-JS-CSS
```

Replace `ComponentName` with any of the component names above (e.g., `Aurora-JS-CSS`, `Beams-JS-CSS`, etc.)

