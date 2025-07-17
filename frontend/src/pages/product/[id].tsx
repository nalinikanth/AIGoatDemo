// @ts-nocheck
import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useDispatch } from 'react-redux';
import MainLayout from '../../components/MainLayout/MainLayout';
import SingleProductRenderer from '../../components/SingleProduct/SingleProductRenderer';
import { useProductSelector } from '../../selectors';
import { fetchProductById } from '../../actions';
import { Row, Comment, Avatar, Form, Input, Skeleton } from 'antd';
import Comments from "../../components/Comments/Comments";
const { TextArea } = Input;
const { Button } = Skeleton;

const isRedTeam = process.env.NEXT_PUBLIC_RED_TEAM_MODE === 'true';

const Product = () => {
  const [isLoading, setLoading] = useState(false);
  const [review, setReview] = useState('');
  const [rating, setRating] = useState(5);
  const [user, setUser] = useState('attacker'); // For demo, default to 'attacker'
  const [malicious, setMalicious] = useState(false);
  const [poisonResult, setPoisonResult] = useState('');
  const [showRecImpact, setShowRecImpact] = useState(false);
  const [recImpact, setRecImpact] = useState('');

  const router = useRouter();
  const { id: productParam } = router.query;
  const productId = productParam ? productParam : null;

  const { currentProduct } = useProductSelector();
  const currentProductId = `${currentProduct?.id ?? ''}`;
  const currentProductName = currentProduct?.name ?? '...';

  const dispatch = useDispatch();

  useEffect(() => {
    if (productId && productId !== currentProductId) {
      setLoading(true);
      dispatch(
          // @ts-ignore
        fetchProductById(productId, () => {
          setLoading(false);
        })
      );
    }
  }, [productId]);

  const handleReviewSubmit = async (e) => {
    e.preventDefault();
    if (isRedTeam && malicious) {
      // Submit to poison-data endpoint
      const res = await fetch('/api/poison-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user, product: currentProduct?.id, rating })
      });
      const data = await res.json();
      setPoisonResult('Malicious review submitted!');
      setShowRecImpact(true);
    } else {
      // ... normal review submission logic ...
    }
  };

  const handleRecImpact = async () => {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user })
    });
    const data = await res.json();
    setRecImpact(JSON.stringify(data, null, 2));
  };

  return (
    <MainLayout title={`AI-Goat - ${currentProductName}`}>
      <SingleProductRenderer
        product={currentProduct}
        loading={isLoading}
        breakpoints={[
          { xl: 10, lg: 10, md: 10, sm: 24, xs: 0 },
          { xl: 14, lg: 14, md: 14, sm: 24, xs: 0 }
        ]}
      />
      <Row
        align="middle"
        justify={"space-around"}
        style={{
          marginTop: 0,
          background: 'white',
          // borderRadius: borderRadiusLG,
          // paddingRight: 200,
          // paddingLeft: 200,
        }}
      >
          {productId && (<Comments productId={Number(productId)} />)}
      </Row>
      <Row
        align="middle"
        justify={"space-around"}
        style={{
          marginTop: 0,
          background: 'white',
          // borderRadius: borderRadiusLG,
          // paddingRight: 200,
          // paddingLeft: 200,
        }}
      >
      </Row>
    </MainLayout>
  );
};

export default Product;
