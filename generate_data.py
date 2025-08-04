"""
Enhanced Dataset Generator for FMCG Promotion Copy Re-use Recommender

This script generates a comprehensive, realistic dataset for FMCG promotional campaigns
with real Indonesian and international brands, enhanced performance metrics, and 
business context.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class FMCGDatasetGenerator:
    """
    Generates realistic FMCG promotional datasets with enhanced features.
    """
    
    def __init__(self):
        """Initialize the dataset generator with realistic brand and category data."""
        
        # Real Indonesian FMCG brands with their typical categories
        self.indonesian_brands = {
            'Wings': ['Detergent', 'Personal Care', 'Home Care'],
            'Mayora': ['Snacks', 'Beverages', 'Confectionery'],
            'Kalbe': ['Health Supplements', 'Dairy', 'Beverages'],
            'Sido Muncul': ['Health Supplements', 'Traditional Medicine', 'Beverages'],
            'Indofood': ['Instant Noodles', 'Snacks', 'Condiments'],
            'ABC': ['Condiments', 'Beverages'],
            'Teh Botol Sosro': ['Beverages'],
            'Chitato': ['Snacks'],
            'Indomie': ['Instant Noodles'],
            'Ultra Milk': ['Dairy'],
            'Wardah': ['Beauty', 'Personal Care'],
            'Mustika Ratu': ['Beauty', 'Personal Care'],
            'Aqua': ['Beverages'],
            'Yakult': ['Dairy', 'Health Supplements']
        }
        
        # International FMCG brands
        self.international_brands = {
            'Unilever': ['Personal Care', 'Home Care', 'Beauty'],
            'P&G': ['Personal Care', 'Home Care', 'Beauty'],
            'Nestlé': ['Dairy', 'Beverages', 'Confectionery', 'Baby Care'],
            'Coca-Cola': ['Beverages'],
            'PepsiCo': ['Beverages', 'Snacks'],
            'Johnson & Johnson': ['Baby Care', 'Personal Care'],
            'L\'Oréal': ['Beauty', 'Personal Care'],
            'Mars': ['Confectionery', 'Snacks'],
            'Mondelez': ['Confectionery', 'Snacks'],
            'Colgate': ['Personal Care'],
            'Nivea': ['Personal Care', 'Beauty'],
            'Oral-B': ['Personal Care'],
            'Pantene': ['Personal Care'],
            'Head & Shoulders': ['Personal Care'],
            'Dove': ['Personal Care', 'Beauty'],
            'Lux': ['Personal Care'],
            'Lifebuoy': ['Personal Care'],
            'Sunsilk': ['Personal Care'],
            'Clear': ['Personal Care'],
            'Vaseline': ['Personal Care']
        }
        
        # Touchpoint channels with realistic distribution
        self.touchpoints = {
            'digital': 0.45,  # 45% of campaigns
            'in-store': 0.30,  # 30% of campaigns
            'flyer': 0.15,     # 15% of campaigns
            'tv': 0.05,        # 5% of campaigns
            'radio': 0.03,     # 3% of campaigns
            'outdoor': 0.02    # 2% of campaigns
        }
        
        # Campaign objectives with performance expectations
        self.campaign_objectives = {
            'awareness': {'lift_range': (0.15, 0.35), 'roi_range': (2.5, 4.0)},
            'conversion': {'lift_range': (0.20, 0.50), 'roi_range': (3.0, 6.0)},
            'retention': {'lift_range': (0.10, 0.25), 'roi_range': (2.0, 3.5)},
            'acquisition': {'lift_range': (0.25, 0.45), 'roi_range': (3.5, 5.5)},
            'upselling': {'lift_range': (0.15, 0.30), 'roi_range': (2.8, 4.5)}
        }
        
        # Budget tiers affecting performance
        self.budget_tiers = {
            'low': {'multiplier': 0.8, 'range': (50, 500)},      # Million IDR
            'medium': {'multiplier': 1.0, 'range': (500, 2000)},
            'high': {'multiplier': 1.3, 'range': (2000, 10000)},
            'premium': {'multiplier': 1.5, 'range': (10000, 50000)}
        }
        
        # Target demographics
        self.demographics = {
            'age_groups': ['18-25', '26-35', '36-45', '46-55', '55+'],
            'income_levels': ['Low', 'Middle', 'Upper-Middle', 'High'],
            'regions': ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Makassar', 'Palembang', 'Tangerang', 'Depok', 'Bekasi']
        }
        
        # Creative elements
        self.creative_elements = {
            'color_schemes': ['Red-Bold', 'Blue-Trust', 'Green-Fresh', 'Yellow-Energy', 'Orange-Warm', 'Purple-Premium', 'Black-Elegant'],
            'imagery_types': ['Product-Focus', 'Lifestyle', 'Family', 'Celebrity', 'Animation', 'Minimalist', 'Festival'],
            'cta_styles': ['Urgent', 'Friendly', 'Professional', 'Playful', 'Authoritative', 'Emotional'],
            'tones': ['Exciting', 'Trustworthy', 'Friendly', 'Premium', 'Fun', 'Serious', 'Caring']
        }
        
        # Promotion types with realistic headline patterns
        self.promotion_patterns = {
            'discount': [
                "DISKON {percent}% - {urgency}!",
                "SALE {percent}% OFF - {product}!",
                "HEMAT {percent}% - {benefit}!",
                "SPECIAL {percent}% DISCOUNT - {timing}!"
            ],
            'bogo': [
                "BELI {qty1} GRATIS {qty2} - {benefit}!",
                "BUY {qty1} GET {qty2} FREE - {product}!",
                "BELI {qty1} DAPAT {qty2} - {urgency}!"
            ],
            'bundle': [
                "PAKET HEMAT - {product1} + {product2}!",
                "BUNDLE DEAL - {benefit}!",
                "COMBO SPECIAL - {value}!"
            ],
            'cashback': [
                "CASHBACK {percent}% - {benefit}!",
                "DAPAT CASHBACK Rp {amount} - {timing}!",
                "CASHBACK SPECIAL - {urgency}!"
            ],
            'free_gift': [
                "GRATIS {gift} - {condition}!",
                "FREE {gift} - {timing}!",
                "BONUS {gift} - {benefit}!"
            ],
            'contest': [
                "MENANG {prize} - {action}!",
                "CONTEST {theme} - {prize}!",
                "KOMPETISI {action} - {reward}!"
            ]
        }
        
    def generate_headline(self, promo_type, brand, category):
        """Generate realistic promotional headline based on type and brand."""
        
        # Select pattern based on promotion type
        if promo_type in self.promotion_patterns:
            pattern = random.choice(self.promotion_patterns[promo_type])
        else:
            pattern = "PROMO SPECIAL {brand} - {benefit}!"
        
        # Fill in pattern variables
        replacements = {
            'percent': random.choice([10, 15, 20, 25, 30, 40, 50, 70]),
            'urgency': random.choice(['Stok Terbatas', 'Hari Ini Saja', 'Terbatas', 'Jangan Terlewat']),
            'product': f"{brand} {category}",
            'benefit': random.choice(['Hemat Lebih', 'Super Value', 'Best Deal', 'Mega Hemat']),
            'timing': random.choice(['Akhir Bulan', 'Weekend Special', 'Flash Sale', 'Limited Time']),
            'qty1': random.choice([1, 2]),
            'qty2': random.choice([1, 2]),
            'product1': brand,
            'product2': random.choice(['Bonus Pack', 'Extra Size', 'Travel Size']),
            'value': random.choice(['Lebih Hemat', 'Double Value', 'Triple Save']),
            'amount': random.choice([5000, 10000, 15000, 25000, 50000]),
            'gift': random.choice(['Tumbler', 'Tas', 'Voucher', 'Sample', 'Sticker']),
            'condition': random.choice(['Pembelian Minimal', 'Member Khusus', 'Pelanggan Setia']),
            'action': random.choice(['Upload Foto', 'Share Post', 'Kirim Video']),
            'prize': random.choice(['Hadiah Jutaan', 'Trip Bali', 'Smartphone', 'Voucher 1 Juta']),
            'theme': random.choice(['Foto Kreativ', 'Video TikTok', 'Story Instagram']),
            'reward': random.choice(['Total 100 Juta', 'Hadiah Mingguan', 'Grand Prize']),
            'brand': brand
        }
        
        # Replace placeholders
        headline = pattern
        for key, value in replacements.items():
            headline = headline.replace(f'{{{key}}}', str(value))
        
        return headline
    
    def generate_performance_metrics(self, objective, budget_tier, touchpoint, brand_tier):
        """Generate realistic performance metrics based on campaign parameters."""
        
        # Base performance from objective
        base_lift_range = self.campaign_objectives[objective]['lift_range']
        base_roi_range = self.campaign_objectives[objective]['roi_range']
        
        # Apply budget tier multiplier
        budget_multiplier = self.budget_tiers[budget_tier]['multiplier']
        
        # Touchpoint effectiveness multipliers
        touchpoint_multipliers = {
            'digital': 1.2,
            'tv': 1.4,
            'in-store': 1.0,
            'flyer': 0.8,
            'radio': 0.9,
            'outdoor': 0.7
        }
        
        # Brand tier multipliers
        brand_multipliers = {
            'premium': 1.3,
            'mass': 1.0,
            'economy': 0.8
        }
        
        # Calculate final multiplier
        total_multiplier = (budget_multiplier * 
                          touchpoint_multipliers.get(touchpoint, 1.0) * 
                          brand_multipliers.get(brand_tier, 1.0))
        
        # Generate metrics with some randomness
        kpi_lift = np.random.uniform(base_lift_range[0], base_lift_range[1]) * total_multiplier
        kpi_roi = np.random.uniform(base_roi_range[0], base_roi_range[1]) * total_multiplier
        
        # Add additional metrics
        ctr = np.random.uniform(0.01, 0.08) * total_multiplier  # Click-through rate
        conversion_rate = np.random.uniform(0.02, 0.15) * total_multiplier
        engagement_rate = np.random.uniform(0.05, 0.25) * total_multiplier
        roas = kpi_roi * np.random.uniform(0.8, 1.2)  # Return on ad spend
        
        # Ensure realistic bounds
        kpi_lift = min(max(kpi_lift, 0.05), 0.80)
        kpi_roi = min(max(kpi_roi, 1.5), 8.0)
        ctr = min(max(ctr, 0.005), 0.12)
        conversion_rate = min(max(conversion_rate, 0.01), 0.25)
        engagement_rate = min(max(engagement_rate, 0.02), 0.40)
        roas = min(max(roas, 1.2), 10.0)
        
        return {
            'kpi_lift': round(kpi_lift, 3),
            'kpi_roi': round(kpi_roi, 1),
            'ctr': round(ctr, 4),
            'conversion_rate': round(conversion_rate, 3),
            'engagement_rate': round(engagement_rate, 3),
            'roas': round(roas, 1)
        }
    
    def generate_sku_code(self, brand, category, variant=""):
        """Generate realistic SKU codes."""
        brand_code = brand[:3].upper()
        category_code = category[:3].upper()
        variant_code = variant[:3].upper() if variant else ""
        number = random.randint(100, 999)
        
        if variant_code:
            return f"{brand_code}-{category_code}-{variant_code}-{number}"
        else:
            return f"{brand_code}-{category_code}-{number}"
    
    def generate_promotional_dataset(self, num_records=200):
        """Generate the main promotional dataset."""
        
        print(f"Generating {num_records} promotional records...")
        
        records = []
        all_brands = {**self.indonesian_brands, **self.international_brands}
        
        for i in range(num_records):
            # Select brand and category
            brand = random.choice(list(all_brands.keys()))
            category = random.choice(all_brands[brand])
            
            # Determine brand tier
            if brand in self.indonesian_brands:
                brand_tier = random.choice(['mass', 'economy']) if random.random() < 0.7 else 'premium'
            else:
                brand_tier = random.choice(['premium', 'mass']) if random.random() < 0.8 else 'economy'
            
            # Campaign parameters
            objective = random.choice(list(self.campaign_objectives.keys()))
            budget_tier = random.choices(
                list(self.budget_tiers.keys()),
                weights=[0.3, 0.4, 0.2, 0.1]  # Most campaigns are low-medium budget
            )[0]
            
            # Select touchpoint based on realistic distribution
            touchpoint = random.choices(
                list(self.touchpoints.keys()),
                weights=list(self.touchpoints.values())
            )[0]
            
            # Promotion type
            promo_type = random.choice(['discount', 'bogo', 'bundle', 'cashback', 'free_gift', 'contest'])
            
            # Generate headline
            headline = self.generate_headline(promo_type, brand, category)
            
            # Performance metrics
            metrics = self.generate_performance_metrics(objective, budget_tier, touchpoint, brand_tier)
            
            # Campaign timing
            start_date = datetime.now() - timedelta(days=random.randint(30, 365))
            period = f"{start_date.year}-Q{(start_date.month-1)//3 + 1}"
            
            # Demographics and targeting
            age_group = random.choice(self.demographics['age_groups'])
            income_level = random.choice(self.demographics['income_levels'])
            region = random.choice(self.demographics['regions'])
            
            # Creative elements
            color_scheme = random.choice(self.creative_elements['color_schemes'])
            imagery_type = random.choice(self.creative_elements['imagery_types'])
            cta_style = random.choice(self.creative_elements['cta_styles'])
            tone = random.choice(self.creative_elements['tones'])
            
            # Budget
            budget_range = self.budget_tiers[budget_tier]['range']
            budget = random.randint(budget_range[0], budget_range[1])
            
            # Duration (days)
            duration = random.choice([7, 14, 21, 30, 45, 60, 90])
            
            # Generate SKU
            sku = self.generate_sku_code(brand, category)
            
            # Description
            description = f"{promo_type.title()} campaign for {brand} {category} targeting {age_group} demographic in {region}"
            
            record = {
                'promo_id': f"PROMO{i+1:03d}",
                'headline': headline,
                'touchpoint': touchpoint,
                'category': category,
                'brand': brand,
                'brand_tier': brand_tier,
                'sku': sku,
                'period': period,
                'campaign_objective': objective,
                'budget_tier': budget_tier,
                'budget_million_idr': budget,
                'duration_days': duration,
                'target_age_group': age_group,
                'target_income_level': income_level,
                'target_region': region,
                'color_scheme': color_scheme,
                'imagery_type': imagery_type,
                'cta_style': cta_style,
                'tone': tone,
                'promo_type': promo_type,
                'kpi_lift': metrics['kpi_lift'],
                'kpi_roi': metrics['kpi_roi'],
                'ctr': metrics['ctr'],
                'conversion_rate': metrics['conversion_rate'],
                'engagement_rate': metrics['engagement_rate'],
                'roas': metrics['roas'],
                'headline_length': len(headline),
                'description': description,
                'description_length': len(description)
            }
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Add performance quartiles
        df['kpi_lift_quartile'] = pd.qcut(df['kpi_lift'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        df['kpi_roi_quartile'] = pd.qcut(df['kpi_roi'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        
        print(f"✅ Generated {len(df)} promotional records")
        return df
    
    def generate_enhanced_sku_dataset(self, promo_df):
        """Generate enhanced SKU master dataset based on promotional data."""
        
        print("Generating enhanced SKU master dataset...")
        
        # Extract unique brand-category combinations from promo data
        brand_categories = promo_df[['brand', 'category', 'brand_tier']].drop_duplicates()
        
        sku_records = []
        
        for _, row in brand_categories.iterrows():
            brand = row['brand']
            category = row['category']
            brand_tier = row['brand_tier']
            
            # Generate multiple SKUs per brand-category
            num_skus = random.randint(2, 5)
            
            for i in range(num_skus):
                # Product variants
                variants = {
                    'Beverages': ['Original', 'Sugar-Free', 'Light', 'Extra', 'Premium'],
                    'Snacks': ['Original', 'BBQ', 'Cheese', 'Spicy', 'Family Pack'],
                    'Personal Care': ['Regular', 'Sensitive', 'Whitening', 'Anti-Aging', 'Natural'],
                    'Dairy': ['Full Cream', 'Low Fat', 'Chocolate', 'Strawberry', 'Vanilla'],
                    'Instant Noodles': ['Original', 'Chicken', 'Beef', 'Seafood', 'Spicy'],
                    'Detergent': ['Regular', 'Anti-Bacterial', 'Softener', 'Concentrated', 'Eco'],
                    'Beauty': ['Day Cream', 'Night Cream', 'Serum', 'Cleanser', 'Moisturizer'],
                    'Home Care': ['Regular', 'Antibacterial', 'Fresh', 'Lemon', 'Lavender']
                }
                
                variant = random.choice(variants.get(category, ['Regular', 'Premium', 'Special']))
                
                # Pack sizes based on category
                pack_sizes = {
                    'Beverages': ['250ml', '350ml', '500ml', '1L', '1.5L'],
                    'Snacks': ['50g', '75g', '100g', '150g', '200g'],
                    'Personal Care': ['100ml', '150ml', '200ml', '250ml', '300ml'],
                    'Dairy': ['125ml', '200ml', '250ml', '500ml', '1L'],
                    'Instant Noodles': ['70g', '85g', '90g', '120g'],
                    'Beauty': ['30ml', '50ml', '100ml', '150ml'],
                    'Home Care': ['500ml', '800ml', '1L', '2L']
                }
                
                pack_size = random.choice(pack_sizes.get(category, ['100g', '200g', '500ml']))
                
                # Generate SKU code
                sku = self.generate_sku_code(brand, category, variant)
                
                # Product name
                product_name = f"{brand} {category} {variant}"
                
                # Price based on brand tier and pack size
                base_prices = {
                    'economy': (5000, 15000),
                    'mass': (10000, 30000),
                    'premium': (25000, 75000)
                }
                
                price_range = base_prices[brand_tier]
                price = random.randint(price_range[0], price_range[1])
                
                # Additional attributes
                launch_year = random.randint(2018, 2024)
                is_flagship = random.choice([True, False]) if brand_tier == 'premium' else False
                
                sku_record = {
                    'sku': sku,
                    'brand': brand,
                    'brand_tier': brand_tier,
                    'category': category,
                    'product_name': product_name,
                    'variant': variant,
                    'pack_size': pack_size,
                    'price_idr': price,
                    'launch_year': launch_year,
                    'is_flagship': is_flagship
                }
                
                sku_records.append(sku_record)
        
        sku_df = pd.DataFrame(sku_records)
        print(f"✅ Generated {len(sku_df)} SKU records")
        return sku_df

def main():
    """Generate enhanced datasets."""
    
    print("🎯 FMCG Enhanced Dataset Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = FMCGDatasetGenerator()
    
    # Generate promotional dataset (Phase 1)
    promo_df = generator.generate_promotional_dataset(num_records=200)
    
    # Generate SKU dataset
    sku_df = generator.generate_enhanced_sku_dataset(promo_df)
    
    # Save datasets
    print("\n💾 Saving datasets...")
    
    # Save enhanced promotional data
    promo_df.to_csv('data/enhanced_promos.csv', index=False)
    print(f"✅ Saved enhanced_promos.csv ({len(promo_df)} records)")
    
    # Save enhanced SKU data
    sku_df.to_csv('data/enhanced_skus.csv', index=False)
    print(f"✅ Saved enhanced_skus.csv ({len(sku_df)} records)")
    
    # Generate summary statistics
    print("\n📊 Dataset Summary:")
    print(f"Promotional Records: {len(promo_df)}")
    print(f"SKU Records: {len(sku_df)}")
    print(f"Brands: {promo_df['brand'].nunique()}")
    print(f"Categories: {promo_df['category'].nunique()}")
    print(f"Touchpoints: {promo_df['touchpoint'].nunique()}")
    print(f"Indonesian Brands: {len([b for b in promo_df['brand'].unique() if b in generator.indonesian_brands])}")
    print(f"International Brands: {len([b for b in promo_df['brand'].unique() if b in generator.international_brands])}")
    
    # Performance summary
    print(f"\nPerformance Metrics:")
    print(f"Average KPI Lift: {promo_df['kpi_lift'].mean():.1%}")
    print(f"Average ROI: {promo_df['kpi_roi'].mean():.1f}x")
    print(f"Average CTR: {promo_df['ctr'].mean():.2%}")
    print(f"Average Conversion Rate: {promo_df['conversion_rate'].mean():.1%}")
    
    print("\n🎉 Enhanced dataset generation completed!")
    print("Ready for use with the FMCG Promotion Copy Re-use Recommender!")

if __name__ == "__main__":
    main()